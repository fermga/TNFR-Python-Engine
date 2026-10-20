"""Full-state modes, capture premises and unavailable continued-contact claims.

Static algebraic states verify the geometry without running a contacted
trajectory or replaying a frozen preparation producer.
"""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_contact_maintenance as owner
from tnfr.physics import _organized_external_interaction as external

_ERROR = Q(1, 10**10)
_EDGES = (
    tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(6) for j in range(9))
    + tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5) for j in range(9))
    + tuple((9 * c + j, 45 + j) for c in range(5) for j in range(9))
)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)


def _assess(**changes):
    return owner._assess_organized_contact_maintenance(
        **(
            dict(form_error_bounds=(_ERROR,) * 6, phase_error_bounds=(_ERROR,) * 6)
            | changes
        )
    )


@pytest.fixture(scope="module")
def geometry():
    return owner._derive_organized_contact_maintenance()


@pytest.fixture(scope="module")
def report():
    return _assess()


def _edge_action(mp, vector, phases=None):
    result = [mp.mpc(0)] * 54
    for left, right in _EDGES:
        weight = 1 if phases is None else mp.cos(phases[right] - phases[left])
        current = weight * (vector[left] - vector[right])
        result[left] += current
        result[right] -= current
    return tuple(result)


def _phase_hessian(mp, phases):
    matrix = mp.zeros(54)
    for left, right in _EDGES:
        weight = mp.cos(phases[right] - phases[left])
        matrix[left, left] += weight
        matrix[right, right] += weight
        matrix[left, right] -= weight
        matrix[right, left] -= weight
    return matrix


def _host_reconstruction_rows(mp, phases):
    """Assemble host torques without supplying or reading a donor phase."""
    torque = [mp.mpf(0)] * 45
    for left, right in _EDGES:
        if left >= 45 or right >= 45:
            continue
        current = mp.sin(phases[right] - phases[left])
        torque[left] += current
        torque[right] -= current
    for slot in range(9):
        angles = tuple(phases[9 * child + slot] for child in range(5))
        design = mp.matrix([[mp.sin(value), -mp.cos(value)] for value in angles])
        gram = design.T * design
        target = mp.matrix([torque[9 * child + slot] for child in range(5)])
        reconstructed = mp.lu_solve(gram, design.T * target)
        yield angles, design, gram, target, reconstructed


@pytest.mark.parametrize("outer_one", (False, True))
@pytest.mark.parametrize("moving_form", (False, True))
def test_contact_convergence_full_law_and_strict_modified_dissipation(
    mp, outer_one, moving_form
):
    """Static edge identities test the proof constants, not convergence in time."""
    tolerance = mp.mpf("1e-165")
    gamma, epsilon = 1 / (1023 * mp.pi), mp.mpf(1) / 8
    scale = gamma**2 / 16
    gaps = (6 * mp.pi / 7,) + (mp.pi / 7,) * 8 if outer_one else (2 * mp.pi / 9,) * 9
    profile = tuple(sum(gaps[:slot], mp.mpf(0)) for slot in range(9))
    phases = tuple(
        profile[slot] + ((child - 2) * mp.pi / 3 if outer_one and child < 5 else 0)
        for child in range(6)
        for slot in range(9)
    )
    # The nonacute control retains all six internal and nine rim periods.
    # Its additive common profile also makes every mixed host period zero.
    for cycle in tuple(tuple(9 * c + j for j in range(9)) for c in range(6)) + tuple(
        tuple(9 * c + j for c in range(5)) for j in range(9)
    ):
        angles = tuple(
            mp.atan2(
                mp.sin(phases[cycle[(i + 1) % len(cycle)]] - phases[node]),
                mp.cos(phases[cycle[(i + 1) % len(cycle)]] - phases[node]),
            )
            for i, node in enumerate(cycle)
        )
        expected = 2 * mp.pi if len(cycle) == 9 or outer_one else 0
        assert abs(sum(angles) - expected) < tolerance
        assert all(abs(angle) < mp.pi for angle in angles)
    if outer_one:
        assert any(mp.cos(phases[b] - phases[a]) < 0 for a, b in _EDGES)

    raw_form = tuple(mp.mpf((7 * index) % 17 - 8) / 19 for index in range(54))
    form = mp.matrix(
        [value - sum(raw_form) / 54 if moving_form else 0 for value in raw_form]
    )
    laplacian, current = mp.zeros(54), mp.zeros(54, 1)
    potential = mp.mpf(0)
    for left, right in _EDGES:
        gap = phases[right] - phases[left]
        potential += 1 - mp.cos(gap)
        current[left] += mp.sin(gap)
        current[right] -= mp.sin(gap)
        laplacian[left, left] += 1
        laplacian[right, right] += 1
        laplacian[left, right] -= 1
        laplacian[right, left] -= 1
    form_rate = (-laplacian * form + gamma * current) / 4
    velocity = gamma * laplacian * form / 4
    acceleration = gamma * laplacian * form_rate / 4
    gradient = -scale * current

    # A gauge-fixed Poisson solve independently supplies P=L^{-1} on 1-perp.
    poisson = laplacian + mp.ones(54) / 54
    mass_velocity = mp.lu_solve(poisson, velocity)
    mass_acceleration = mp.lu_solve(poisson, acceleration)
    assert abs(sum(form)) < tolerance
    assert abs(sum(velocity)) < tolerance
    assert abs(sum(mass_velocity)) < tolerance
    assert max(abs(value) for value in mass_velocity - gamma * form / 4) < tolerance
    assert (
        max(abs(value) for value in mass_acceleration + velocity / 4 + gradient)
        < tolerance
    )
    storage = sum((form[b] - form[a]) ** 2 for a, b in _EDGES) / 2 + potential
    energy = (velocity.T * mass_velocity)[0] / 2 + scale * potential
    assert abs(energy - scale * storage) < tolerance
    velocity_squared = (velocity.T * velocity)[0]
    energy_rate = (mass_velocity.T * acceleration + gradient.T * velocity)[0]
    assert abs(energy_rate + velocity_squared / 4) < tolerance

    def modified_energy_on_tangent(parameter):
        """Differentiate a straight tangent, without taking an evolution step."""
        varied_velocity = velocity + parameter * acceleration
        varied_mass = mass_velocity + parameter * mass_acceleration
        varied_gradient = mp.zeros(54, 1)
        varied_potential = mp.mpf(0)
        for left, right in _EDGES:
            gap = (
                phases[right]
                - phases[left]
                + parameter * (velocity[right] - velocity[left])
            )
            varied_potential += 1 - mp.cos(gap)
            force = scale * mp.sin(gap)
            varied_gradient[left] -= force
            varied_gradient[right] += force
        return (
            (varied_velocity.T * varied_mass)[0] / 2
            + scale * varied_potential
            + epsilon * (varied_gradient.T * varied_velocity)[0]
        )

    direct_rate = mp.diff(modified_energy_on_tangent, 0)
    hessian_quadratic = scale * sum(
        mp.cos(phases[b] - phases[a]) * (velocity[b] - velocity[a]) ** 2
        for a, b in _EDGES
    )
    gradient_loss = (gradient.T * laplacian * gradient)[0]
    expected_rate = -velocity_squared / 4 + epsilon * (
        hessian_quadratic - (gradient.T * laplacian * velocity)[0] / 4 - gradient_loss
    )
    bound = -17 * velocity_squared / 128 - gradient_loss / 16
    assert abs(direct_rate - expected_rate) < tolerance
    assert direct_rate <= bound + tolerance
    if outer_one or moving_form:
        assert direct_rate < 0
        assert bound < 0
    else:
        assert max(abs(value) for value in current) < tolerance
        assert abs(direct_rate) < tolerance
    if outer_one and not moving_form:
        assert energy_rate == 0
        assert abs(direct_rate + epsilon * gradient_loss) < tolerance


def test_donor_reconstruction_uniform_host_twist_fails_the_circle_condition(mp):
    """Zero host torque has a unique linear solution, but it is not a phase."""
    tolerance = mp.mpf("1e-165")
    phases = tuple(
        2 * mp.pi * (slot - 4) / 9 + 2 * mp.pi * (child - 2) / 5
        for child in range(5)
        for slot in range(9)
    )
    for _, design, gram, target, reconstructed in _host_reconstruction_rows(mp, phases):
        assert max(abs(value) for value in target) < tolerance
        assert max(abs(value) for value in gram - mp.eye(2) * 5 / 2) < tolerance
        assert max(abs(value) for value in reconstructed) < tolerance
        assert max(abs(value) for value in design * reconstructed - target) < tolerance
        circle_defect = (reconstructed.T * reconstructed)[0] - 1
        assert abs(circle_defect + 1) < tolerance
        assert circle_defect < -mp.mpf("0.99")


def test_donor_reconstruction_covariance_does_not_remove_the_full_residual(mp):
    """A single fine phase change gives an exact nonzero least-squares defect."""
    tolerance = mp.mpf("1e-165")
    alpha, perturbation, origin = 2 * mp.pi / 9, mp.mpf(1) / 13, mp.mpf(2) / 7
    phases = tuple(
        alpha * (slot - 4)
        + 2 * mp.pi * (child - 2) / 5
        + (perturbation if (child, slot) == (0, 4) else 0)
        for child in range(5)
        for slot in range(9)
    )
    rotation = mp.matrix(
        [[mp.cos(origin), -mp.sin(origin)], [mp.sin(origin), mp.cos(origin)]]
    )
    rows = tuple(_host_reconstruction_rows(mp, phases))
    shifted = tuple(_host_reconstruction_rows(mp, tuple(h + origin for h in phases)))
    for row, moved in zip(rows, shifted):
        angles, design, gram, target, reconstructed = row
        _, new_design, new_gram, new_target, new_reconstructed = moved
        determinant = sum(
            mp.sin(angles[left] - angles[right]) ** 2
            for left in range(5)
            for right in range(left + 1, 5)
        )
        assert determinant > 0
        assert abs(mp.det(gram) - determinant) < tolerance
        assert max(abs(value) for value in new_target - target) < tolerance
        assert max(abs(value) for value in new_design - design * rotation.T) < tolerance
        assert (
            max(abs(value) for value in new_gram - rotation * gram * rotation.T)
            < tolerance
        )
        assert (
            max(abs(value) for value in new_reconstructed - rotation * reconstructed)
            < tolerance
        )
        residual = design * reconstructed - target
        new_residual = new_design * new_reconstructed - new_target
        assert max(abs(value) for value in new_residual - residual) < tolerance
        assert max(abs(value) for value in design.T * residual) < tolerance

    # At the preceding slot only child zero acquires a host torque t.
    _, design, gram, target, reconstructed = rows[3]
    current = mp.sin(alpha + perturbation) - mp.sin(alpha)
    assert current > 0
    assert abs(target[0] - current) < tolerance
    assert max(abs(target[child]) for child in range(1, 5)) < tolerance
    assert max(abs(value) for value in gram - mp.eye(2) * 5 / 2) < tolerance
    residual = design * reconstructed - target
    residual_squared = (residual.T * residual)[0]
    assert abs(residual_squared - 3 * current**2 / 5) < tolerance
    assert residual_squared > 0


def test_host_reorganization_uniform_rim_has_a_finite_current_floor(mp):
    """A static fine perturbation retains the floor with arbitrary donor phases."""
    tolerance, epsilon = mp.mpf("1e-165"), mp.mpf(1) / 200
    origins = tuple(2 * mp.pi * (child - 2) / 5 for child in range(5))
    profile = tuple(
        2 * mp.pi * (slot - 4) / 9 + mp.mpf((2 * slot + 1) % 7 - 3) / 50
        for slot in range(9)
    )
    donor_origins = tuple(mp.mpf(2 * slot - 8) / 7 for slot in range(9))
    host_reference = tuple(
        profile[slot] + origins[child] for child in range(5) for slot in range(9)
    )
    host = tuple(
        value + epsilon * ((3 * (index // 9) + 5 * (index % 9)) % 7 - 3) / 3
        for index, value in enumerate(host_reference)
    )
    assert all(
        abs(mp.exp(mp.j * actual) - mp.exp(mp.j * reference)) <= epsilon
        for actual, reference in zip(host, host_reference)
    )
    phases = host + tuple(
        value + origin for value, origin in zip(profile, donor_origins)
    )
    current = [mp.mpf(0)] * 54
    for left, right in _EDGES:
        value = mp.sin(phases[right] - phases[left])
        current[left] += value
        current[right] -= value
    floor = 1 / mp.sqrt(2) - 9 * epsilon
    assert floor > 0
    for slot in range(9):
        internal = mp.sin(profile[(slot + 1) % 9] - profile[slot]) + mp.sin(
            profile[(slot - 1) % 9] - profile[slot]
        )
        ideal = tuple(
            internal + mp.sin(donor_origins[slot] - origin) for origin in origins
        )
        actual = tuple(current[9 * child + slot] for child in range(5))
        assert (
            abs(sum(value**2 for value in ideal) - 5 * internal**2 - mp.mpf(5) / 2)
            < tolerance
        )
        assert (
            max(abs(value - reference) for value, reference in zip(actual, ideal))
            <= 9 * epsilon
        )
        assert max(abs(value) for value in ideal) >= 1 / mp.sqrt(2)
        assert max(abs(value) for value in actual) >= floor


def test_host_reorganization_conditioned_pair_bound_on_a_nonstationary_control(mp):
    """Supplied chord bounds test the envelope, without inferring stationarity."""
    tolerance, epsilon = mp.mpf("1e-165"), mp.mpf("1e-5")
    origins = tuple((child - 2) * mp.pi / 3 for child in range(5))
    design = mp.matrix([[1, mp.cos(angle), -mp.sin(angle)] for angle in origins])
    expected_gram = mp.matrix([[5, 1, 0], [1, 2, 0], [0, 0, 3]])
    assert max(abs(value) for value in design.T * design - expected_gram) < tolerance
    sigma = mp.sqrt((7 - mp.sqrt(13)) / 2)
    coefficient = 18 * mp.sqrt(5) / sigma
    gaps = (6 * mp.pi / 7,) + (mp.pi / 7,) * 8
    profile = tuple(sum(gaps[:slot], mp.mpf(0)) for slot in range(9))
    donor_origins = tuple(epsilon * (slot - 4) / 4 for slot in range(9))
    beta = tuple(mp.exp(mp.j * angle) for angle in donor_origins)
    host_reference = tuple(
        profile[slot] + origins[child] for child in range(5) for slot in range(9)
    )
    host = tuple(
        value + epsilon * ((3 * (index // 9) + 5 * (index % 9)) % 7 - 3) / 3
        for index, value in enumerate(host_reference)
    )
    assert all(
        abs(mp.exp(mp.j * actual) - mp.exp(mp.j * reference)) <= epsilon
        for actual, reference in zip(host, host_reference)
    )
    assert all(abs(value - beta[0]) <= coefficient * epsilon for value in beta)
    phases = host + tuple(
        value + origin for value, origin in zip(profile, donor_origins)
    )
    current = [mp.mpf(0)] * 54
    for left, right in _EDGES:
        value = mp.sin(phases[right] - phases[left])
        current[left] += value
        current[right] -= value
    assert max(abs(value) for value in current[:45]) > mp.mpf("0.5")
    rim = mp.matrix(
        [
            mp.sin(origins[(child + 1) % 5] - angle)
            + mp.sin(origins[(child - 1) % 5] - angle)
            for child, angle in enumerate(origins)
        ]
    )
    rows = []
    for slot in range(9):
        internal = mp.sin(profile[(slot + 1) % 9] - profile[slot]) + mp.sin(
            profile[(slot - 1) % 9] - profile[slot]
        )
        coefficients = mp.matrix([internal, beta[slot].imag, beta[slot].real])
        actual = mp.matrix([current[9 * child + slot] for child in range(5)])
        error = actual - design * coefficients - rim
        assert max(abs(value) for value in error) <= 9 * epsilon
        rows.append((coefficients, actual, error))
    for coefficients, actual, error in rows:
        delta = coefficients - rows[0][0]
        residual_difference = actual - rows[0][1]
        assert (
            max(
                abs(value)
                for value in design * delta - residual_difference + error - rows[0][2]
            )
            < tolerance
        )
        assert (
            mp.norm(delta)
            <= (18 * mp.sqrt(5) * epsilon + mp.norm(residual_difference)) / sigma
        )

    # The fixed pair has no internal-edge cost, even on the nonacute profile.
    reference = host_reference + tuple(value + donor_origins[0] for value in profile)
    child_phasors = tuple(mp.exp(mp.j * angle) for angle in origins) + (beta[0],)
    real = tuple(value.real for value in child_phasors for _ in range(9))
    imaginary = tuple(value.imag for value in child_phasors for _ in range(9))
    pair_values = []
    for state in (reference, phases):
        pair_values.append(
            sum(
                mp.cos(state[right] - state[left])
                * (
                    (real[right] - real[left]) ** 2
                    + (imaginary[right] - imaginary[left]) ** 2
                )
                for left, right in _EDGES
            )
        )
    assert pair_values[0] <= -mp.mpf(9) / 2
    error_bound = (540 + 180 * coefficient) * epsilon
    assert abs(pair_values[1] - pair_values[0]) <= error_bound
    margin = mp.mpf(9) / 2 - error_bound
    assert margin > 0
    assert pair_values[1] <= -margin
    rayleigh = []
    for vector in (real, imaginary):
        centered = tuple(value - sum(vector) / 54 for value in vector)
        quadratic = sum(
            value * action
            for value, action in zip(centered, _edge_action(mp, centered, phases))
        ).real
        rayleigh.append(quadratic / sum(value**2 for value in centered))
    assert min(rayleigh) <= -margin / 54


@pytest.mark.parametrize("positive_children", ((0,), (0, 2)))
@pytest.mark.parametrize("antipodal_donor", (False, True))
def test_host_reorganization_antipodal_boundary_has_a_negative_child_cut(
    mp, positive_children, antipodal_donor
):
    """Marked-cut boundary states do not bypass the common-host obstruction."""
    tolerance = mp.mpf("1e-165")
    profile = tuple(2 * mp.pi * (slot - 4) / 9 for slot in range(9))
    donor_origins = tuple(
        mp.mpf(slot - 4) / 20 + (mp.pi if antipodal_donor else 0) for slot in range(9)
    )
    phases = tuple(
        profile[slot] + (0 if child in positive_children else mp.pi)
        for child in range(5)
        for slot in range(9)
    ) + tuple(value + origin for value, origin in zip(profile, donor_origins))
    crossings = sum(
        (child in positive_children) != ((child + 1) % 5 in positive_children)
        for child in range(5)
    )
    assert crossings in (2, 4)
    donor_sum = sum(mp.cos(origin) for origin in donor_origins)
    assert (donor_sum < 0) == antipodal_donor
    quadratics = []
    for positive in (True, False):
        children = tuple(
            child for child in range(5) if (child in positive_children) == positive
        )
        cut = tuple(mp.mpf(index // 9 in children) for index in range(54))
        centered = tuple(value - sum(cut) / 54 for value in cut)
        expected = -9 * crossings + (1 if positive else -1) * len(children) * donor_sum
        for vector in (cut, centered):
            quadratic = sum(
                value * action
                for value, action in zip(vector, _edge_action(mp, vector, phases))
            ).real
            assert abs(quadratic - expected) < tolerance
        quadratics.append(quadratic)
    assert min(quadratics) <= -9 * crossings
    assert min(quadratics) <= -18


def test_donor_schur_singular_block_has_an_explicit_full_negative_witness(mp):
    """The winding-one carrier has a singular donor block, not a stable target."""
    tolerance = mp.mpf("1e-165")
    phases = tuple(
        2 * mp.pi * (j - 4) / 9 + (2 * mp.pi * (c - 2) / 5 if c < 5 else 0)
        for c in range(6)
        for j in range(9)
    )
    matrix = _phase_hessian(mp, phases)
    cross, donor = matrix[:45, 45:], matrix[45:, 45:]
    null = mp.matrix([1] * 9)
    # Internal cycle currents telescope; the five spoke cosines sum to zero.
    assert max(abs(value) for value in donor * null) < tolerance
    expected_cross_gram = mp.eye(9) * mp.mpf(5) / 2
    assert (
        max(abs(value) for value in cross.T * cross - expected_cross_gram) < tolerance
    )
    coupling = cross * null
    norm_squared = (coupling.T * coupling)[0]
    assert abs(norm_squared - mp.mpf(45) / 2) < tolerance
    full = tuple(-coupling / 10) + tuple(null)
    vector = tuple(value - sum(full) / 54 for value in full)
    quadratic = sum(
        value * action
        for value, action in zip(vector, _edge_action(mp, vector, phases))
    ).real
    assert quadratic <= -norm_squared / 10 + tolerance
    assert quadratic < -2


def test_donor_schur_positive_block_does_not_admit_host_relaxation_at_blind_point(mp):
    """A static nonstationary control retains a negative relaxed host direction."""
    tolerance = mp.mpf("1e-165")
    gaps = (mp.pi / 3,) * 7 + (-mp.pi / 6,) * 2
    profile = tuple(sum(gaps[:j], mp.mpf(0)) for j in range(9))
    phases = tuple(
        profile[j] + ((c - 2) * mp.pi / 3 if c < 5 else 0)
        for c in range(6)
        for j in range(9)
    )
    matrix = _phase_hessian(mp, phases)
    host, cross, donor = matrix[:45, :45], matrix[:45, 45:], matrix[45:, 45:]
    internal = mp.zeros(9)
    for j, gap in enumerate(gaps):
        next_slot = (j + 1) % 9
        weight = mp.cos(gap)
        assert weight > 0
        internal[j, j] += weight
        internal[next_slot, next_slot] += weight
        internal[j, next_slot] -= weight
        internal[next_slot, j] -= weight
    # Positive internal edge weights make internal PSD, hence donor >= I.
    assert max(abs(value) for value in donor - internal - mp.eye(9)) < tolerance
    donor_one = mp.matrix([1] * 9)
    assert max(abs(value) for value in donor * donor_one - donor_one) < tolerance
    inverse = donor**-1
    reduced = host - cross * inverse * cross.T
    assert max(abs(value) for value in reduced * mp.matrix([1] * 45)) < tolerance

    direction = mp.matrix([int(i < 9) for i in range(45)])
    forcing = cross.T * direction
    relaxed = -inverse * forcing
    assert abs((direction.T * host * direction)[0] + mp.mpf(9) / 2) < tolerance
    assert max(abs(value) for value in forcing - donor_one / 2) < tolerance
    assert max(abs(value) for value in relaxed + donor_one / 2) < tolerance
    expected = -mp.mpf(27) / 4
    assert abs((direction.T * reduced * direction)[0] - expected) < tolerance
    full = tuple(direction) + tuple(relaxed)
    centered = tuple(value - sum(full) / 54 for value in full)
    for vector in (full, centered):
        quadratic = sum(
            value * action
            for value, action in zip(vector, _edge_action(mp, vector, phases))
        ).real
        assert abs(quadratic - expected) < tolerance

    # A separate fine vector retains donor residuals in the completed square.
    arbitrary = tuple(mp.mpf((7 * i + 3) % 19 - 9) / 23 for i in range(54))
    host_vector, donor_vector = mp.matrix(arbitrary[:45]), mp.matrix(arbitrary[45:])
    residual = donor_vector + inverse * cross.T * host_vector
    donor_cost = (residual.T * donor * residual)[0]
    assert donor_cost > 0
    completed = (host_vector.T * reduced * host_vector)[0] + donor_cost
    direct = sum(
        value * action
        for value, action in zip(arbitrary, _edge_action(mp, arbitrary, phases))
    ).real
    assert abs(completed - direct) < tolerance


def test_host_relaxation_retains_eight_relative_modes_and_full_curvature(mp):
    """The aligned outer-zero control checks local curvature, not temporal rates."""
    tolerance = mp.mpf("1e-165")
    alpha = 2 * mp.pi / 9
    phases = tuple(alpha * (slot - 4) for _ in range(6) for slot in range(9))
    matrix = _phase_hessian(mp, phases)
    host, cross, donor = matrix[:45, :45], matrix[:45, 45:], matrix[45:, 45:]
    cycle = mp.zeros(9)
    for slot in range(9):
        neighbor = (slot + 1) % 9
        cycle[slot, slot] += 1
        cycle[neighbor, neighbor] += 1
        cycle[slot, neighbor] -= 1
        cycle[neighbor, slot] -= 1
    common_inverse = (mp.cos(alpha) * cycle + mp.eye(9)) ** -1
    response = mp.matrix(
        [[common_inverse[row % 9, slot] for slot in range(9)] for row in range(45)]
    )
    assert max(abs(value) for value in host * response + cross) < tolerance
    coupling = -cross.T * response
    reduced = donor - coupling
    assert max(abs(value) for value in reduced * mp.matrix([1] * 9)) < tolerance

    host_positive = donor_positive = donor_neutral = 0
    for mode in range(9):
        internal = tuple(mp.exp(mp.j * alpha * mode * slot) for slot in range(9))
        stiffness = mp.cos(alpha) * (2 - 2 * mp.cos(alpha * mode))
        for rim_mode in range(5):
            vector = mp.matrix(
                [
                    mp.exp(2 * mp.j * mp.pi * rim_mode * child / 5) * internal[slot]
                    for child in range(5)
                    for slot in range(9)
                ]
            )
            eigenvalue = 1 + stiffness + 2 - 2 * mp.cos(2 * mp.pi * rim_mode / 5)
            assert eigenvalue >= 1
            assert (
                max(abs(value) for value in host * vector - eigenvalue * vector)
                < tolerance
            )
            host_positive += 1
        donor_vector = mp.matrix(internal)
        expected_response = mp.matrix(internal * 5) / (stiffness + 1)
        assert (
            max(abs(value) for value in response * donor_vector - expected_response)
            < tolerance
        )
        eigenvalue = stiffness * (stiffness + 6) / (stiffness + 1)
        assert (
            max(
                abs(value)
                for value in reduced * donor_vector - eigenvalue * donor_vector
            )
            < tolerance
        )
        if mode == 0:
            assert eigenvalue == 0
            donor_neutral += 1
        else:
            assert eigenvalue > 0
            donor_positive += 1
    assert (host_positive, donor_positive, donor_neutral) == (45, 8, 1)

    # Complete the full quadratic form, retaining all 45 host fiber directions.
    arbitrary = tuple(mp.mpf((7 * index + 3) % 19 - 9) / 23 for index in range(54))
    host_vector, donor_vector = mp.matrix(arbitrary[:45]), mp.matrix(arbitrary[45:])
    residual = host_vector - response * donor_vector
    fiber_cost = (residual.T * host * residual)[0]
    assert fiber_cost > 0
    completed = fiber_cost + (donor_vector.T * reduced * donor_vector)[0]
    direct = sum(
        value * action
        for value, action in zip(arbitrary, _edge_action(mp, arbitrary, phases))
    ).real
    assert abs(completed - direct) < tolerance

    # Match the older donor-relaxed K to the new host-relaxed representation.
    donor_inverse = donor**-1
    old_reduced = host - cross * donor_inverse * cross.T
    pullback = response.T * old_reduced * response
    coupling_form = coupling - coupling * donor_inverse * coupling
    reduced_form = reduced - reduced * donor_inverse * reduced
    assert max(abs(value) for value in pullback - coupling_form) < tolerance
    assert max(abs(value) for value in pullback - reduced_form) < tolerance
    transverse = mp.matrix(
        [
            (mp.mpf(slot - 4) / 9) * (int(child == 0) - int(child == 1))
            for child in range(5)
            for slot in range(9)
        ]
    )
    assert max(abs(value) for value in cross.T * transverse) < tolerance
    assert (
        max(abs(value) for value in response.T * old_reduced * transverse) < tolerance
    )
    assert (transverse.T * host * transverse)[0] > 0
    assert abs((transverse.T * (old_reduced - host) * transverse)[0]) < tolerance


def test_donor_schur_aligned_control_has_all_44_positive_relative_modes(mp):
    """Check static reduced curvature; these values are not temporal eigenvalues."""
    tolerance = mp.mpf("1e-165")
    alpha = 2 * mp.pi / 9
    phases = tuple(alpha * (j - 4) for _ in range(6) for j in range(9))
    matrix = _phase_hessian(mp, phases)
    host, cross, donor = matrix[:45, :45], matrix[:45, 45:], matrix[45:, 45:]
    cycle = mp.zeros(9)
    for j in range(9):
        next_slot = (j + 1) % 9
        cycle[j, j] += 1
        cycle[next_slot, next_slot] += 1
        cycle[j, next_slot] -= 1
        cycle[next_slot, j] -= 1
    expected_donor = mp.cos(alpha) * cycle + 5 * mp.eye(9)
    assert max(abs(value) for value in donor - expected_donor) < tolerance
    inverse = donor**-1
    reduced = host - cross * inverse * cross.T
    positive = neutral = 0
    for internal_mode in range(9):
        internal = tuple(
            mp.exp(2 * mp.j * mp.pi * internal_mode * j / 9) for j in range(9)
        )
        stiffness = mp.cos(alpha) * (2 - 2 * mp.cos(alpha * internal_mode))
        assert stiffness + 5 > 0
        for rim_mode in range(5):
            vector = mp.matrix(
                [
                    mp.exp(2 * mp.j * mp.pi * rim_mode * c / 5) * internal[j]
                    for c in range(5)
                    for j in range(9)
                ]
            )
            if rim_mode == 0:
                expected = stiffness * (stiffness + 6) / (stiffness + 5)
                expected_relaxed = mp.matrix(internal) * 5 / (stiffness + 5)
            else:
                expected = stiffness + 1 + 2 - 2 * mp.cos(2 * mp.pi * rim_mode / 5)
                expected_relaxed = mp.zeros(9, 1)
            action = reduced * vector
            assert max(abs(value) for value in action - expected * vector) < tolerance
            relaxed = -inverse * cross.T * vector
            assert max(abs(value) for value in relaxed - expected_relaxed) < tolerance
            if (internal_mode, rim_mode) == (0, 0):
                assert expected == 0
                neutral += 1
            else:
                assert expected > 0
                positive += 1
    assert (positive, neutral) == (44, 1)


def test_stationary_resultants_do_not_make_unrestricted_stress_positive(mp):
    """Static controls distinguish tangent curvature from unrestricted stress."""
    alpha = 2 * mp.pi / 9
    tolerance = mp.mpf("1e-165")
    degrees = [0] * 54
    phases = tuple(alpha * (j - 4) for _ in range(6) for j in range(9))
    phasors = tuple(mp.exp(mp.j * phase) for phase in phases)
    resultants = [mp.mpc(0)] * 54
    for a, b in _EDGES:
        resultants[a] += phasors[b]
        resultants[b] += phasors[a]
        degrees[a] += 1
        degrees[b] += 1
    radial = tuple(mp.conj(z) * resultant for z, resultant in zip(phasors, resultants))
    assert degrees == [5] * 45 + [7] * 9
    assert all(abs(value.imag) < tolerance for value in radial)
    lower_bound = 5 - 2 * mp.sqrt(6)
    assert lower_bound > mp.mpf("0.1")
    for i, value in enumerate(radial):
        expected = 2 * mp.cos(alpha) + (3 if i < 45 else 5)
        assert abs(value.real - expected) < tolerance
        assert value.real > lower_bound
        assert abs(phasors[i] - resultants[i] / abs(resultants[i])) < tolerance
        # Stationarity is H*z=0, where H=diag(r)-Adj; it is not H>=0.
        assert abs(value.real * phasors[i] - resultants[i]) < tolerance
        cosine_square_sum = sum(
            mp.cos(phases[b] - phases[a]) ** 2 for a, b in _EDGES if i in (a, b)
        )
        assert cosine_square_sum >= 1
        assert value.real**2 + cosine_square_sum <= 10 * value.real
    potential = sum(1 - mp.cos(phases[b] - phases[a]) for a, b in _EDGES)
    stress_on_one = tuple(value.real - degree for value, degree in zip(radial, degrees))
    assert all(
        abs(value - (2 * mp.cos(alpha) - 2)) < tolerance for value in stress_on_one
    )
    assert abs(sum(stress_on_one) + 2 * potential) < tolerance
    assert sum(stress_on_one) < 0
    # All phase-Hessian weights are positive on this connected support.
    # The independent full-mode control below verifies the relative spectrum.
    assert min(mp.cos(phases[b] - phases[a]) for a, b in _EDGES) > 0
    vector = tuple(mp.mpf(2 * i - 53) / 108 for i in range(54))
    curvature = sum(
        v * value for v, value in zip(vector, _edge_action(mp, vector, phases))
    ).real
    form_curvature = sum(
        v * value for v, value in zip(vector, _edge_action(mp, vector))
    ).real
    assert curvature >= mp.cos(alpha) * form_curvature - tolerance
    assert form_curvature > 0

    antipodal = tuple(
        phase + (mp.pi if i >= 45 else 0) for i, phase in enumerate(phases)
    )
    anti_z = tuple(mp.exp(mp.j * phase) for phase in antipodal)
    anti_resultants = [mp.mpc(0)] * 54
    for a, b in _EDGES:
        anti_resultants[a] += anti_z[b]
        anti_resultants[b] += anti_z[a]
    anti_radial = tuple(mp.conj(z) * value for z, value in zip(anti_z, anti_resultants))
    assert all(abs(value.imag) < tolerance for value in anti_radial)
    for value in anti_radial[45:]:
        assert abs(value.real - (2 * mp.cos(alpha) - 5)) < tolerance
        assert value.real < 0
    donor_cut = tuple(mp.mpf(i >= 45) - mp.mpf(1) / 6 for i in range(54))
    negative = sum(
        value * action
        for value, action in zip(donor_cut, _edge_action(mp, donor_cut, antipodal))
    ).real
    assert abs(negative + 45) < tolerance


def test_static_slot_shift_secant_and_endpoint_hessian_defect(mp):
    """The nonstationary blind point is a secant control, not a target verdict."""
    gaps = (Q(1, 6),) * 7 + (Q(-1, 12),) * 2
    profile = tuple(sum(gaps[:j], Q(0)) for j in range(9))
    raw = tuple(
        profile[j] + (Q(c - 2, 6) if c < 5 else 0) for c in range(6) for j in range(9)
    )
    mean = sum(raw, Q(0)) / 54
    turns = tuple(value - mean for value in raw)
    shifted = tuple(
        turns[9 * c + (j + 1) % 9] + int(j == 8) - Q(1, 9)
        for c in range(6)
        for j in range(9)
    )
    assert sum(turns) == sum(shifted) == 0
    assert tuple(shifted[i] - turns[i] for i in range(54)) == tuple(
        gaps[j] - Q(1, 9) for _ in range(6) for j in range(9)
    )
    branches = tuple(-int(a % 9 == 8) for a, _ in _EDGES[:54])
    branches += tuple(-int(a // 9 == 4) for a, _ in _EDGES[54:99])
    branches += (0,) * 45
    gap_rows = []
    for state in (turns, shifted):
        edge_gaps = tuple(
            state[b] - state[a] - branch for (a, b), branch in zip(_EDGES, branches)
        )
        gap_rows.append(edge_gaps)
        assert all(-Q(1, 2) < value < Q(1, 2) for value in edge_gaps[:99])
        assert all(sum(edge_gaps[9 * c : 9 * c + 9]) == 1 for c in range(6))
        assert all(
            sum(edge_gaps[54 + 9 * c + j] for c in range(5)) == 1 for j in range(9)
        )
        for c in range(5):
            for j in range(9):
                mixed_square = (
                    edge_gaps[9 * c + j]
                    + edge_gaps[54 + 9 * c + (j + 1) % 9]
                    - edge_gaps[9 * ((c + 1) % 5) + j]
                    - edge_gaps[54 + 9 * c + j]
                )
                assert mixed_square == 0
    assert gap_rows[0][:9] == gaps
    assert gap_rows[1][:9] == gaps[1:] + gaps[:1]
    assert gap_rows[0][54:] == gap_rows[1][54:]

    phases = tuple(2 * mp.pi * _mp(mp, value) for value in turns)
    plus = tuple(2 * mp.pi * _mp(mp, value) for value in shifted)
    direction = tuple(after - before for before, after in zip(phases, plus))
    current_rows = []
    for state in (phases, plus):
        current = [mp.mpf(0)] * 54
        for a, b in _EDGES:
            value = mp.sin(state[b] - state[a])
            current[a] += value
            current[b] -= value
        current_rows.append(current)
    tolerance = mp.mpf("1e-165")
    assert abs(current_rows[0][1] - mp.sqrt(3) / 2) < tolerance
    secant = mp.mpf(0)
    endpoint_edges = mp.mpf(0)
    for old, new in zip(*gap_rows):
        a, b = 2 * mp.pi * _mp(mp, old), 2 * mp.pi * _mp(mp, new)
        secant += (mp.sin(b) - mp.sin(a)) * (b - a)
        endpoint_edges += (mp.cos(a) + mp.cos(b)) * (b - a) ** 2 / 2
    expected_secant = 3 * mp.pi * (mp.sqrt(3) + 1)
    expected_endpoint = 3 * mp.pi**2 * (mp.sqrt(3) + 1) / 4
    assert abs(secant - expected_secant) < tolerance
    assert secant > 0
    current_pairing = sum(
        h * (after - before)
        for h, before, after in zip(direction, current_rows[0], current_rows[1])
    )
    assert abs(current_pairing + secant) < tolerance
    quadratic = tuple(
        sum(
            h * action
            for h, action in zip(direction, _edge_action(mp, direction, state))
        ).real
        for state in (phases, plus)
    )
    assert all(abs(value - expected_endpoint) < tolerance for value in quadratic)
    average = sum(quadratic) / 2
    assert abs(average - endpoint_edges) < tolerance
    midpoint = tuple((a + b) / 2 for a, b in zip(phases, plus))
    midpoint_quadratic = sum(
        h * action
        for h, action in zip(direction, _edge_action(mp, direction, midpoint))
    ).real
    # Every edge with a nonzero directional gap changes by exactly pi/2.
    assert (
        abs(secant - mp.sin(mp.pi / 4) / (mp.pi / 4) * midpoint_quadratic) < tolerance
    )
    assert abs(average - mp.cos(mp.pi / 4) * midpoint_quadratic) < tolerance
    assert abs((average - secant) - expected_secant * (mp.pi / 4 - 1)) < tolerance
    assert average - secant < 0


@pytest.mark.parametrize("antipodal", (False, True))
def test_every_full_spatial_mode_against_independent_edge_matrices(
    mp, geometry, antipodal
):
    """Apply the actual 54-node Laplacian and Hessian to complex Fourier modes."""
    phases = tuple(
        2 * mp.pi * (j - 4) / 9 + (mp.pi if antipodal and c == 5 else 0)
        for c in range(6)
        for j in range(9)
    )
    gamma = 1 / (1023 * mp.pi)
    stable = unstable = neutral = 0
    for mode in geometry.modes:
        if mode.outer_mode == "common":
            outer = (mp.mpc(1),) * 6
        elif mode.outer_mode == "hub":
            outer = (mp.mpc(1),) * 5 + (mp.mpc(-5),)
        else:
            frequency = int(mode.outer_mode[4:])
            outer = tuple(mp.exp(2j * mp.pi * frequency * c / 5) for c in range(5))
            outer += (mp.mpc(0),)
        vector = tuple(
            outer[c] * mp.exp(2j * mp.pi * mode.internal_mode * j / 9)
            for c in range(6)
            for j in range(9)
        )
        norm = sum(abs(value) ** 2 for value in vector)
        laplacian_action = _edge_action(mp, vector)
        hessian_action = _edge_action(mp, vector, phases)
        laplacian = (
            sum(mp.conj(v) * action for v, action in zip(vector, laplacian_action)).real
            / norm
        )
        hessian = (
            sum(mp.conj(v) * action for v, action in zip(vector, hessian_action)).real
            / norm
        )
        for value, left, right in zip(vector, laplacian_action, hessian_action):
            assert abs(left - laplacian * value) < mp.mpf("1e-165")
            assert abs(right - hessian * value) < mp.mpf("1e-165")
        hessian_box = (
            mode.antipodal_hessian_bounds if antipodal else mode.aligned_hessian_bounds
        )
        determinant_box = (
            mode.antipodal_determinant_bounds
            if antipodal
            else mode.aligned_determinant_bounds
        )
        if (mode.internal_mode, mode.outer_mode) == (0, "common"):
            assert mode.laplacian_bounds == I(0)
            assert hessian_box == I(0)
            assert determinant_box == I(0)
            neutral += 2
            continue
        _contains(mp, mode.laplacian_bounds, laplacian)
        _contains(mp, hessian_box, hessian)
        _contains(mp, mode.trace_bounds, -laplacian / 4)
        _contains(mp, determinant_box, gamma**2 * laplacian * hessian / 16)
        # The two roots use both complete tangent rows, including quarter time.
        discriminant = laplacian**2 - 4 * gamma**2 * laplacian * hessian
        roots = (
            (-laplacian + mp.sqrt(discriminant)) / 8,
            (-laplacian - mp.sqrt(discriminant)) / 8,
        )
        stable += sum(value.real < 0 for value in roots)
        unstable += sum(value.real > 0 for value in roots)
    assert neutral == geometry.conserved_mean_modes == 2
    if antipodal:
        assert stable == geometry.antipodal_relative_stable_modes == 97
        assert unstable == geometry.antipodal_relative_unstable_modes == 9
    else:
        assert stable == geometry.aligned_relative_stable_modes == 106
        assert unstable == 0


def test_geometry_covers_actual_product_and_retains_neutral_modes(geometry):
    assert len(geometry.modes) == 54
    assert len({(m.internal_mode, m.outer_mode) for m in geometry.modes}) == 54
    assert geometry.aligned_stability_certified
    assert geometry.antipodal_instability_certified
    assert geometry.near_carrier_obstruction_certified
    neutral = next(
        mode
        for mode in geometry.modes
        if (mode.internal_mode, mode.outer_mode) == (0, "common")
    )
    assert neutral.trace_bounds == I(0)
    assert all(
        mode.antipodal_determinant_bounds.hi < 0
        for mode in geometry.modes
        if mode.outer_mode == "hub"
    )
    assert geometry.interaction_geometry.external_phase_origin_bounds == I(0)
    assert geometry.antipodal_donor_phase_origin_bounds.lo > 3


@pytest.mark.parametrize("beta", (Q(-2), Q(0), Q(7, 5)))
@pytest.mark.parametrize("fine_amplitude", (Q(0), Q(1, 100)))
def test_negative_hessian_witness_and_fine_defect_without_equilibrium_claim(
    mp, geometry, beta, fine_amplitude
):
    phases, offsets = [], []
    for c in range(6):
        for j in range(9):
            carrier = 2 * mp.pi * (c - 2) / 5 if c < 5 else _mp(mp, beta)
            fine = _mp(mp, fine_amplitude) * mp.sin(2 * mp.pi * j / 9 + c)
            offsets.append(carrier + fine)
            phases.append(2 * mp.pi * (j - 4) / 9 + carrier + fine)
    vectors = (
        tuple(mp.sin(value) for value in offsets),
        tuple(mp.cos(value) for value in offsets),
    )
    total = mp.mpf(0)
    for vector in vectors:
        mean = sum(vector) / 54
        projected = tuple(value - mean for value in vector)
        unprojected = sum(
            mp.cos(phases[b] - phases[a]) * (vector[b] - vector[a]) ** 2
            for a, b in _EDGES
        )
        quadratic = sum(
            mp.cos(phases[b] - phases[a]) * (projected[b] - projected[a]) ** 2
            for a, b in _EDGES
        )
        assert abs(quadratic - unprojected) < mp.mpf("1e-165")
        total += quadratic
    defect = sum(1 - mp.cos(offsets[b] - offsets[a]) for a, b in _EDGES[:54])
    exact_threshold = 45 * (7 - 3 * mp.sqrt(5)) / 8
    _contains(mp, geometry.internal_phase_defect_threshold_bounds, exact_threshold)
    assert defect < exact_threshold
    assert total < 2 * (defect - exact_threshold) < 0
    # These static witnesses are generally moving states, not fictitious equilibria.
    currents = [mp.mpf(0)] * 54
    for a, b in _EDGES:
        current = mp.sin(phases[b] - phases[a])
        currents[a] += current
        currents[b] -= current
    assert max(abs(value) for value in currents) > mp.mpf("0.1")


def test_actual_control_capture_keeps_source_history_and_scope(report):
    assert report.source_certified
    assert report.control_capture_certified
    assert report.control_identity_retention_certified
    assert report.control_convergence_certified
    assert report.admitted
    assert report.unresolved_conditions == ()
    assert report.relative_phase_origin == 0
    assert report.relative_preparation_dimension == 96
    assert report.protocol_revision == "4c4addf1"
    assert report.control_initial_excess_upper_bound < Q(1, 100)
    assert report.control_initial_excess_upper_bound < Q(2, 10**12)
    assert report.control_capture_barrier_lower_bound > Q(6, 25)
    assert report.control_attachment_work_upper_bound < Q(1, 10**12)
    assert all(value > 0 for _, value in report.sufficient_margins)
    assert report.outer_one_initial_defect_domain_certified
    assert report.outer_one_initial_defect_margin > 1
    assert report.outer_one_maintenance_status.startswith("unavailable_")
    assert report.other_origin_maintenance_status.startswith("unavailable_")
    assert "no_all_future_positive_individual_inner_gap_claim" in report.scope
    assert (
        "initial_defect_domain_inclusion_does_not_prove_it_is_retained" in report.scope
    )
    assert (
        report.external_interaction.observation_protocol.total_absolute_event_work_upper_bound
        > 0
    )


def test_sharper_control_bounds_derive_from_whole_word_not_generic_state_class(report):
    source = report.external_interaction
    host, observation = source.host_acquisition, source.observation_protocol
    gamma = report.geometry.interaction_geometry.gamma_bounds.hi
    duration = observation.total_elapsed_scaled_time
    b = host.full_form_error_upper_bound + gamma * duration
    e = host.full_phase_error_upper_bound + 2 * gamma * b * duration
    assert report.control_host_form_upper_bound == b
    assert report.control_host_phase_offset_upper_bound == e
    assert b < source.observation_form_upper_bound
    assert (
        e < host.full_phase_error_upper_bound + source.carried_phase_drift_upper_bound
    )
    assert (
        report.control_form_excess_upper_bound
        == 288 * report.control_joined_form_upper_bound**2
    )
    assert (
        report.control_phase_excess_upper_bound
        == 288 * report.control_joined_phase_offset_upper_bound**2
    )
    assert report.control_initial_excess_upper_bound == (
        report.control_form_excess_upper_bound + report.control_phase_excess_upper_bound
    )
    assert (
        report.internal_boundary_excess_bounds is host.internal_boundary_excess_bounds
    )
    assert report.control_capture_barrier_lower_bound == min(
        host.internal_boundary_excess_bounds.lo, 1
    )


def test_fine_state_storage_and_actual_contact_work_fit_complete_bounds(mp, report):
    """Independent node perturbations exercise all three kinds of joined edges."""
    b = _mp(mp, report.control_joined_form_upper_bound)
    e = _mp(mp, report.control_joined_phase_offset_upper_bound)
    forms = tuple(b * mp.sin(2 * mp.pi * i / 54) for i in range(54))
    errors = tuple(e * mp.cos(2 * mp.pi * 7 * i / 54) for i in range(54))
    target = tuple(2 * mp.pi * (i % 9 - 4) / 9 for i in range(54))
    actual_excess = mp.mpf(0)
    for a, z in _EDGES:
        angle = target[z] - target[a]
        actual_excess += (forms[z] - forms[a]) ** 2 / 2
        actual_excess += mp.cos(angle) - mp.cos(angle + errors[z] - errors[a])
    assert 0 <= actual_excess <= _mp(mp, report.control_initial_excess_upper_bound)
    # Attachment uses the sharper host and donor radii, rather than the joined box.
    host_b = _mp(mp, report.control_host_form_upper_bound)
    host_e = _mp(mp, report.control_host_phase_offset_upper_bound)
    donor_radius = mp.mpf("1e-70")
    work = mp.mpf(0)
    for a, z in _EDGES[99:]:
        host_form = host_b * mp.sin(a)
        donor_form = donor_radius * mp.cos(z)
        host_phase = host_e * mp.cos(a)
        donor_phase = donor_radius * mp.sin(z)
        work += (host_form - donor_form) ** 2 / 2
        work += 1 - mp.cos(host_phase - donor_phase)
    assert 0 < work <= _mp(mp, report.control_attachment_work_upper_bound)


def test_unavailable_obstruction_does_not_erase_independent_control_capture(
    monkeypatch, report
):
    monkeypatch.setattr(
        owner,
        "_assess_organized_external_interaction",
        lambda **_: report.external_interaction,
    )
    monkeypatch.setattr(
        owner,
        "_derive_organized_contact_maintenance",
        lambda: replace(report.geometry, near_carrier_obstruction_certified=False),
    )
    result = _assess()
    assert not result.outer_one_initial_defect_domain_certified
    assert result.control_capture_certified
    assert result.control_identity_retention_certified
    assert result.outer_one_maintenance_status.startswith("unavailable_")


def test_unavailable_target_stability_abstains_without_losing_geometry_scope(
    monkeypatch, report
):
    monkeypatch.setattr(
        owner,
        "_assess_organized_external_interaction",
        lambda **_: report.external_interaction,
    )
    monkeypatch.setattr(
        owner,
        "_derive_organized_contact_maintenance",
        lambda: replace(report.geometry, aligned_stability_certified=False),
    )
    result = _assess()
    assert result.source_certified
    assert not result.control_capture_certified
    assert not result.control_identity_retention_certified
    assert not result.control_convergence_certified
    assert "aligned_target_not_certified" in result.unresolved_conditions


def test_source_rebuilt_once_and_old_fixed_origin_verdict_is_not_consumed(
    monkeypatch, report
):
    source = replace(
        report.external_interaction,
        admitted=False,
        comparison_certified=False,
        identity_certified=False,
        work_certified=False,
        observed_contrast_bounds=I(-1),
    )
    calls = []

    def rebuild(**kwargs):
        calls.append(kwargs)
        return source

    monkeypatch.setattr(owner, "_assess_organized_external_interaction", rebuild)
    result = _assess()
    assert calls == [
        dict(form_error_bounds=(_ERROR,) * 6, phase_error_bounds=(_ERROR,) * 6)
    ]
    assert result.external_interaction is source
    assert result.control_capture_certified
    assert result.admitted


@pytest.mark.parametrize(
    "flag_index,condition",
    (
        (0, "host_acquisition_not_certified"),
        (1, "paired_host_control_not_certified"),
        (2, "external_recovery_not_certified"),
        (3, "preceding_sequential_observation_not_certified"),
        (4, "actual_hosts_outside_observation_class"),
    ),
)
def test_missing_actual_source_premise_abstains_in_every_capture_field(
    monkeypatch, report, flag_index, condition
):
    flags = list(report.external_interaction.source_admission_flags)
    flags[flag_index] = False
    source = replace(
        report.external_interaction,
        source_admission_flags=tuple(flags),
        source_certified=False,
    )
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", lambda **_: source
    )
    result = _assess()
    assert not result.source_certified
    assert not result.control_capture_certified
    assert not result.control_identity_retention_certified
    assert not result.control_convergence_certified
    assert not result.outer_one_initial_defect_domain_certified
    assert not result.admitted
    assert condition in result.unresolved_conditions
    assert result.geometry.aligned_stability_certified


@pytest.mark.parametrize("boundary", (I(-1), I(0), I(Q(1, 10**15))))
def test_rebuilt_boundary_must_dominate_actual_excess(monkeypatch, report, boundary):
    host = replace(
        report.external_interaction.host_acquisition,
        internal_boundary_excess_bounds=boundary,
    )
    source = replace(report.external_interaction, host_acquisition=host)
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", lambda **_: source
    )
    result = _assess()
    assert not result.control_capture_certified
    assert not result.control_identity_retention_certified
    assert "control_capture_boundary" in result.unresolved_conditions


@pytest.mark.parametrize("parameter", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize(
    "invalid",
    (
        True,
        float("nan"),
        float("inf"),
        Decimal("0.0"),
        Q(-1, 10**100),
        _ERROR + Q(1, 10**100),
    ),
)
def test_invalid_primitive_allowance_rejected_before_source_rebuild(
    monkeypatch, parameter, invalid
):
    def forbidden(**_):
        pytest.fail("source construction occurred before primitive admission")

    monkeypatch.setattr(external, "_assess_organized_outer_winding", forbidden)
    values = [_ERROR] * 6
    values[-1] = invalid
    with pytest.raises((TypeError, ValueError)):
        _assess(**{parameter: values})


@pytest.mark.parametrize("parameter", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("length", (0, 5, 7))
def test_wrong_source_arity_rejected(monkeypatch, parameter, length):
    def forbidden(**_):
        pytest.fail("source construction occurred before vector admission")

    monkeypatch.setattr(external, "_assess_organized_outer_winding", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{parameter: (_ERROR,) * length})


def test_no_cached_report_input_or_mutable_verdict(report):
    with pytest.raises(TypeError):
        _assess(report=report)
    with pytest.raises(FrozenInstanceError):
        report.admitted = False
