"""Exact rational algebra probes for the paired-ring local composition proof.

The actual prepared equilibrium uses c=cos(2*pi/5), s=sin(2*pi/5) and
inverse_pi=1/pi. These are irrational. This instrument checks a supplied rational
coefficient family; it does not replace those constants in the engine or
certify their exact trigonometric values. The analytic theorem, including
nonlinear failure of the resulting observation, belongs to
theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-composition.
No trajectory, fitted parameter, new law or effective physical node is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
import json

from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_product as product,
    exact_symmetric_semidefinite,
)
from tnfr.mathematics.linear_observation import (
    LinearObservation,
    derive_linear_observation,
)


Matrix = tuple[tuple[Q, ...], ...]


@dataclass(frozen=True)
class LocalCompositionAnalysis:
    """Rational matrix identities, not an irrational-equilibrium certificate."""

    ring_cosine: Q
    inverse_pi: Q
    epi_weight: Q
    phase_weight: Q
    storage_scale: Q
    generator: Matrix
    output_rows: Matrix
    natural_rows: Matrix
    natural_lift: Matrix
    natural_generator: Matrix
    output_map: Matrix
    hidden_directions: Matrix
    offset_directions: Matrix
    odd_directions: Matrix
    hidden_output_rates: Matrix
    realization: LinearObservation
    exact_identity_checks: tuple[str, ...]
    ideal_trigonometric_coefficients: bool = False
    scope: str = (
        "Exact rational coefficient-family identities for one supplied single-bridge "
        "paired-C5 joint linearization. Actual c=cos(2*pi/5), inverse_pi=1/pi "
        "require the separate analytic proof. Fixed unit capacity and support; "
        "no nonlinear closure, trajectory, autonomous nesting or physical identity."
    )


def _rational(value, name, *, allow_zero=False):
    if type(value) not in (int, Q):
        raise TypeError(f"{name} must be an exact int or Fraction, not a float")
    result = Q(value)
    if result < 0 or (not allow_zero and not result):
        raise ValueError(
            f"{name} must be {'nonnegative' if allow_zero else 'positive'}"
        )
    return result


def _transpose(matrix):
    return tuple(zip(*matrix, strict=True))


def _blocks(matrix):
    """Use identical ordered geometry for form and phase observations."""
    width = len(matrix[0])
    zero = (Q(0),) * width
    return tuple(row + zero for row in matrix) + tuple(zero + row for row in matrix)


def _zero(matrix):
    return not any(value for row in matrix for value in row)


def _paired_ring_geometry(c):
    """One unit-support/phase-Hessian assembly for the fixed ordered graph."""
    b = [[Q(0) for _ in range(10)] for _ in range(10)]
    k = [[Q(0) for _ in range(10)] for _ in range(10)]
    degree = [0] * 10
    strength = [Q(0)] * 10
    edges = [
        (offset + i, offset + (i + 1) % 5, c) for offset in (0, 5) for i in range(5)
    ] + [(0, 5, Q(1))]
    for i, j, cosine in edges:
        for matrix, weight in ((b, Q(1)), (k, cosine)):
            matrix[i][i] += weight
            matrix[j][j] += weight
            matrix[i][j] -= weight
            matrix[j][i] -= weight
        degree[i] += 1
        degree[j] += 1
        strength[i] += cosine
        strength[j] += cosine
    return (
        tuple(map(tuple, b)),
        tuple(map(tuple, k)),
        tuple(degree),
        tuple(strength),
    )


def analyze_local_composition(
    *,
    ring_cosine,
    inverse_pi,
    epi_weight=Q(1, 2),
    phase_weight=Q(1, 2),
    storage_scale=Q(1),
) -> LocalCompositionAnalysis:
    """Check the fixed six-output study on declared exact rational coefficients.

    Graph: rings 0..4 and 5..9 with sole bridge (0,5). The full state order
    is ten form deviations then ten phase deviations. Per species the three
    outputs are mean(left)-mean(right), port0-mean(left), port5-mean(right).
    Five natural coordinates add each ring's near-pair minus far-pair average.
    The two global offsets are removed; relative regional offsets are retained.

    Positive c, inverse_pi, w, beta and nonnegative e describe an algebraic
    family, not tunable physical constants or a newly admitted engine model.
    Fixed exact analytic coefficients and the nonlinear boundary have their
    own theorem. Rank is computed exactly with the shared invariant-row owner.
    """
    c = _rational(ring_cosine, "ring_cosine")
    rho = _rational(inverse_pi, "inverse_pi")
    e = _rational(epi_weight, "epi_weight", allow_zero=True)
    w = _rational(phase_weight, "phase_weight")
    beta = _rational(storage_scale, "storage_scale")
    b, k, degree, strength = _paired_ring_geometry(c)
    generator = tuple(
        tuple(-e * entry / degree[i] for entry in b[i])
        + tuple(-w * rho * entry / strength[i] for entry in k[i])
        for i in range(10)
    ) + tuple(
        tuple(w * rho * entry / (beta * strength[i]) for entry in b[i]) + (Q(0),) * 10
        for i in range(10)
    )
    mean_left = tuple(Q(1, 5) if i < 5 else Q(0) for i in range(10))
    mean_right = tuple(Q(1, 5) if i >= 5 else Q(0) for i in range(10))
    rows = (
        tuple(a - b for a, b in zip(mean_left, mean_right, strict=True)),
        tuple(Q(i == 0) - mean_left[i] for i in range(10)),
        tuple(Q(i == 5) - mean_right[i] for i in range(10)),
        tuple(
            Q(1, 2) if i in (1, 4) else -Q(1, 2) if i in (2, 3) else Q(0)
            for i in range(10)
        ),
        tuple(
            Q(1, 2) if i in (6, 9) else -Q(1, 2) if i in (7, 8) else Q(0)
            for i in range(10)
        ),
    )
    outputs, natural = _blocks(rows[:3]), _blocks(rows)
    # Lift with means +/-mu/2 and each ring's three even shape values:
    # port=m+p, near=m-p/4+z/2, far=m-p/4-z/2.
    lift = []
    for i in range(10):
        ring, position = divmod(i, 5)
        row = [Q(0)] * 5
        row[0] = Q(1, 2) if ring == 0 else -Q(1, 2)
        row[1 + ring] = Q(1) if position == 0 else -Q(1, 4)
        row[3 + ring] = (
            Q(0) if position == 0 else Q(1, 2) if position in (1, 4) else -Q(1, 2)
        )
        lift.append(tuple(row))
    natural_lift = _blocks(tuple(lift))
    cj = product(natural, generator)
    natural_generator = product(cj, natural_lift)
    output_map = product(outputs, natural_lift)
    eta_left = tuple(map(Q, (0, 1, -1, -1, 1, 0, 0, 0, 0, 0)))
    eta_right = (Q(0),) * 5 + eta_left[:5]
    hidden = _blocks((eta_left, eta_right))
    offsets = _blocks(((Q(1),) * 10,))
    odd = _blocks(
        tuple(
            tuple(Q(i == left) - Q(i == right) for i in range(10))
            for left, right in ((1, 4), (2, 3), (6, 9), (7, 8))
        )
    )
    hidden_rates = product(product(outputs, generator), _transpose(hidden))
    realization = derive_linear_observation(generator, outputs)
    identity = tuple(tuple(Q(i == j) for j in range(10)) for i in range(10))
    checks = {
        "C T=I": product(natural, natural_lift) == identity,
        "C J=G C": cj == product(natural_generator, natural),
        "O=D C": outputs == product(output_map, natural),
        "global offsets stationary": _zero(product(generator, _transpose(offsets))),
        "global offsets unobserved": _zero(product(natural, _transpose(offsets))),
        "odd directions unobserved": _zero(product(natural, _transpose(odd))),
        "odd directions remain unobserved at first variation": _zero(
            product(cj, _transpose(odd))
        ),
        "initial hidden directions unobserved": _zero(
            product(outputs, _transpose(hidden))
        ),
        "six outputs need ten linear coordinates": realization.rank_progression
        == (6, 10, 10),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"paired-ring composition identities failed: {failed}")
    return LocalCompositionAnalysis(
        c,
        rho,
        e,
        w,
        beta,
        generator,
        outputs,
        natural,
        natural_lift,
        natural_generator,
        output_map,
        hidden,
        offsets,
        odd,
        hidden_rates,
        realization,
        tuple(checks),
    )


@dataclass(frozen=True)
class StateRateObstruction:
    """Static rational coefficient probe for equal state/rate, unequal future.

    State vectors contain form followed by phase DEVIATIONS from the prepared
    winding-one lock, never absolute zero phases. Rates and accelerations are
    the exact algebraic jet at those preparations for the supplied coefficient
    family. Actual trigonometric coefficients require the separate proof.
    """

    composition: LocalCompositionAnalysis
    ring_sine: Q
    even_amplitude: Q
    odd_amplitude: Q
    state_minus: tuple[Q, ...]
    state_plus: tuple[Q, ...]
    rate_minus: tuple[Q, ...]
    rate_plus: tuple[Q, ...]
    acceleration_minus: tuple[Q, ...]
    acceleration_plus: tuple[Q, ...]
    projected_state_minus: tuple[Q, ...]
    projected_state_plus: tuple[Q, ...]
    projected_rate_minus: tuple[Q, ...]
    projected_rate_plus: tuple[Q, ...]
    projected_acceleration_minus: tuple[Q, ...]
    projected_acceleration_plus: tuple[Q, ...]
    scalar_gap_expected: Q
    scalar_gap_actual: Q
    form_storage_minus: Q
    form_storage_plus: Q
    loss_minus: Q
    loss_plus: Q
    obstruction_established: bool
    exact_identity_checks: tuple[str, ...]
    ideal_trigonometric_coefficients: bool = False
    phase_coordinates: str = "deviations_about_prepared_winding_one_lock"
    scope: str = (
        "Exact supplied rational coefficient-family static jet on fixed single-bridge "
        "paired C5 with unit capacity. Zero phase coordinates mean deviations from "
        "the prepared lock. No trajectory, physical identification or rational "
        "replacement of ideal cos(2*pi/5), sin(2*pi/5), pi is certified. A zero "
        "amplitude gives an inconclusive degenerate pair, not a closure theorem."
    )


def analyze_state_rate_obstruction(
    *,
    ring_cosine,
    ring_sine,
    inverse_pi,
    epi_weight=Q(1, 2),
    phase_weight=Q(1, 2),
    storage_scale=Q(1),
    even_amplitude=Q(1, 32),
    odd_amplitude=Q(1, 64),
) -> StateRateObstruction:
    """Check one exact equal-(Cz,CF) pair with different coarse acceleration.

    Use x_minus=epsilon*eta_L-delta*(e1-e4) and x_plus with the opposite
    odd sign, at the identical prepared phase lock. Both amplitudes are
    nonnegative rational inputs; zero is admitted as a no-obstruction control.
    Positive c, sine, inverse_pi, w, beta and nonnegative e define a rational
    algebraic probe, not an asserted realizable trigonometric phase table.

    At this lock F=J*(x,0), but DF*F includes the phase-metric derivative:
    Hdot=-pi*S*F_phase, with S_ij the oriented neighbor sine coefficients.
    Therefore phase acceleration adds k*q*inverse_pi*(S*F_phase)/strength**2
    to J*F, where k=w/beta. This is the same law's static derivative, not a
    replacement linear evolution or a derivative of binary64 arithmetic.
    """
    sine = _rational(ring_sine, "ring_sine")
    epsilon = _rational(even_amplitude, "even_amplitude", allow_zero=True)
    delta = _rational(odd_amplitude, "odd_amplitude", allow_zero=True)
    composition = analyze_local_composition(
        ring_cosine=ring_cosine,
        inverse_pi=inverse_pi,
        epi_weight=epi_weight,
        phase_weight=phase_weight,
        storage_scale=storage_scale,
    )
    c, rho = composition.ring_cosine, composition.inverse_pi
    e, w, beta = (
        composition.epi_weight,
        composition.phase_weight,
        composition.storage_scale,
    )
    b, _, degree, strength = _paired_ring_geometry(c)
    phase_scale = w / beta

    def mv(matrix, vector):
        return tuple(
            sum((a * b for a, b in zip(row, vector, strict=True)), Q(0))
            for row in matrix
        )

    def prepare(sign):
        x = tuple(
            epsilon * even + sign * delta * odd
            for even, odd in zip(
                (0, 1, -1, -1, 1, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, -1, 0, 0, 0, 0, 0),
                strict=True,
            )
        )
        state = x + (Q(0),) * 10
        q = mv(b, x)
        rate = mv(composition.generator, state)
        linear = mv(composition.generator, rate)
        # Ring edges point i->i+1 with positive sine; the bridge sine is zero.
        sine_action = tuple(
            sine
            * (
                rate[10 + 5 * (i // 5) + (i + 1) % 5]
                - rate[10 + 5 * (i // 5) + (i - 1) % 5]
            )
            for i in range(10)
        )
        acceleration = linear[:10] + tuple(
            linear[10 + i]
            + phase_scale * q[i] * rho * sine_action[i] / strength[i] ** 2
            for i in range(10)
        )
        energy = sum((xi * qi for xi, qi in zip(x, q, strict=True)), Q(0)) / 2
        loss = e * sum((q[i] ** 2 / degree[i] for i in range(10)), Q(0))
        return state, rate, acceleration, energy, loss

    minus, plus = prepare(-1), prepare(1)
    cm, cp = (
        tuple(mv(composition.natural_rows, value) for value in state[:3])
        for state in (minus, plus)
    )
    # C phase entries 5 and 6 sum to theta0 - mean(theta_right).
    actual = cp[2][5] + cp[2][6] - cm[2][5] - cm[2][6]
    expected = (
        -8 * phase_scale**2 * epsilon * delta * sine * rho**2 / (c * (1 + 2 * c) ** 2)
    )
    checks = {
        "same ten-coordinate state": cm[0] == cp[0],
        "same ten-coordinate instantaneous rate": cm[1] == cp[1],
        "same form storage": minus[3] == plus[3],
        "same instantaneous storage loss": minus[4] == plus[4],
        "predicted coarse acceleration gap": actual == expected,
        "nonzero amplitudes give distinct coarse acceleration": (
            not epsilon or not delta or actual < 0
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"state-and-rate obstruction identities failed: {failed}")
    return StateRateObstruction(
        composition=composition,
        ring_sine=sine,
        even_amplitude=epsilon,
        odd_amplitude=delta,
        state_minus=minus[0],
        state_plus=plus[0],
        rate_minus=minus[1],
        rate_plus=plus[1],
        acceleration_minus=minus[2],
        acceleration_plus=plus[2],
        projected_state_minus=cm[0],
        projected_state_plus=cp[0],
        projected_rate_minus=cm[1],
        projected_rate_plus=cp[1],
        projected_acceleration_minus=cm[2],
        projected_acceleration_plus=cp[2],
        scalar_gap_expected=expected,
        scalar_gap_actual=actual,
        form_storage_minus=minus[3],
        form_storage_plus=plus[3],
        loss_minus=minus[4],
        loss_plus=plus[4],
        obstruction_established=bool(actual),
        exact_identity_checks=tuple(checks),
    )


@dataclass(frozen=True)
class LocalMemoryAnalysis:
    """Complete visible/hidden split of one rational joint tangent model.

    C,T,P,U are respectively visible projection/lift and hidden projection/lift.
    The combined ten visible and eight hidden coordinates reconstruct all
    twenty fine deviations modulo two common offsets. This is a coordinate
    decomposition, not additional compression, a nonlinear memory closure or
    a diffusion positivity statement. Energy matrices include their factor
    one half: E=y.T*visible_energy*y+h.T*hidden_energy*h.
    """

    composition: LocalCompositionAnalysis
    visible_projection: Matrix
    visible_lift: Matrix
    hidden_projection: Matrix
    hidden_lift: Matrix
    visible_generator: Matrix
    hidden_generator: Matrix
    quotient_projector: Matrix
    visible_energy: Matrix
    hidden_energy: Matrix
    visible_energy_rate: Matrix
    hidden_energy_rate: Matrix
    visible_dissipation: Matrix
    hidden_dissipation: Matrix
    exact_identity_checks: tuple[str, ...]
    ideal_trigonometric_coefficients: bool = False
    scope: str = (
        "Complete 18-coordinate common-offset quotient of a supplied 20-coordinate "
        "rational paired-ring tangent model. Ten visible plus eight hidden variables "
        "are retained. Exact split and quadratic energy identities supply neither "
        "nonlinear closure, numerical remainder constants, a positive memory kernel, "
        "an ODE trajectory nor physical identity. Ideal trigonometric coefficients "
        "and nonlinear approximation orders require their separate analytic proof."
    )


def analyze_local_memory(
    *,
    ring_cosine,
    inverse_pi,
    epi_weight=Q(1, 2),
    phase_weight=Q(1, 2),
    storage_scale=Q(1),
) -> LocalMemoryAnalysis:
    """Derive the full exact visible/hidden decomposition without a solver.

    Hidden rows are half-differences at near and far reflection pairs, ordered
    form left-near/left-far/right-near/right-far, then phase in the same order.
    The lift has opposite unit entries at each pair. Both species lose only
    their global arithmetic offset. The phase entries remain deviations from
    the prepared winding-one lock.

    Input coefficients follow analyze_local_composition's rational-family
    admission. Signed exchange permits oscillatory hidden motion, and zero
    EPI weight has zero quadratic loss; no diffusion-memory positivity or
    nonlinear forgetting theorem is inferred from this static algebra.
    """
    composition = analyze_local_composition(
        ring_cosine=ring_cosine,
        inverse_pi=inverse_pi,
        epi_weight=epi_weight,
        phase_weight=phase_weight,
        storage_scale=storage_scale,
    )
    c, t = composition.natural_rows, composition.natural_lift
    p = tuple(tuple(value / 2 for value in row) for row in composition.odd_directions)
    u = _transpose(composition.odd_directions)
    j, g = composition.generator, composition.natural_generator
    ah = product(product(p, j), u)
    b, phase_hessian, degree, _ = _paired_ring_geometry(composition.ring_cosine)
    zero10 = (Q(0),) * 10
    energy = tuple(tuple(value / 2 for value in row) + zero10 for row in b) + tuple(
        zero10 + tuple(composition.storage_scale * value / 2 for value in row)
        for row in phase_hessian
    )
    loss_form = product(
        b,
        tuple(
            tuple(composition.epi_weight * value / degree[i] for value in row)
            for i, row in enumerate(b)
        ),
    )
    loss = tuple(row + zero10 for row in loss_form) + (zero10 + zero10,) * 10

    def add(left, right):
        return tuple(
            tuple(a + b for a, b in zip(row_a, row_b, strict=True))
            for row_a, row_b in zip(left, right, strict=True)
        )

    def congruence(matrix, lift):
        return product(product(_transpose(lift), matrix), lift)

    def identity(size):
        return tuple(tuple(Q(i == k) for k in range(size)) for i in range(size))

    quotient = _blocks(
        tuple(tuple(Q(i == k) - Q(1, 10) for k in range(10)) for i in range(10))
    )
    ey, eh = congruence(energy, t), congruence(energy, u)
    dy, dh = congruence(loss, t), congruence(loss, u)
    ey_rate = add(product(_transpose(g), ey), product(ey, g))
    eh_rate = add(product(_transpose(ah), eh), product(eh, ah))
    checks = {
        "C T=I10": product(c, t) == identity(10),
        "P U=I8": product(p, u) == identity(8),
        "C U=0": _zero(product(c, u)),
        "P T=0": _zero(product(p, t)),
        "T C+U P=common-offset quotient": add(product(t, c), product(u, p)) == quotient,
        "C J=G C": product(c, j) == product(g, c),
        "P J=Ah P": product(p, j) == product(ah, p),
        "P J T=0": _zero(product(product(p, j), t)),
        "C J U=0": _zero(product(product(c, j), u)),
        "visible and hidden quadratic energy cross term vanishes": _zero(
            product(product(_transpose(t), energy), u)
        ),
        "visible and hidden energy reconstruct full energy": add(
            congruence(ey, c), congruence(eh, p)
        )
        == energy,
        "visible quadratic energy is positive definite": exact_symmetric_semidefinite(
            ey, strict=True
        ),
        "hidden quadratic energy is positive definite": exact_symmetric_semidefinite(
            eh, strict=True
        ),
        "visible energy rate equals negative form dissipation": _zero(add(ey_rate, dy)),
        "hidden energy rate equals negative form dissipation": _zero(add(eh_rate, dh)),
        "visible dissipation is nonnegative": exact_symmetric_semidefinite(dy),
        "hidden dissipation is nonnegative": exact_symmetric_semidefinite(dh),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"local memory split identities failed: {failed}")
    return LocalMemoryAnalysis(
        composition=composition,
        visible_projection=c,
        visible_lift=t,
        hidden_projection=p,
        hidden_lift=u,
        visible_generator=g,
        hidden_generator=ah,
        quotient_projector=quotient,
        visible_energy=ey,
        hidden_energy=eh,
        visible_energy_rate=ey_rate,
        hidden_energy_rate=eh_rate,
        visible_dissipation=dy,
        hidden_dissipation=dh,
        exact_identity_checks=tuple(checks),
    )


def main():
    """Print one rational algebra check, explicitly separate from the theorem."""
    report = analyze_local_composition(ring_cosine=Q(1, 3), inverse_pi=Q(1, 3))
    print(
        json.dumps(
            {
                "study": "paired_C5_single_bridge_local_composition_rational_probe",
                "ring_cosine": "1/3",
                "inverse_pi": "1/3",
                "ideal_trigonometric_coefficients": report.ideal_trigonometric_coefficients,
                "rank_progression": report.realization.rank_progression,
                "exact_identity_checks": report.exact_identity_checks,
                "scope": report.scope,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
