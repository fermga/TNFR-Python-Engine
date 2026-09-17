"""Exact cycle gap transport, winding phase response and joint nodal bounds.

The gap observer uses a simple cycle with ``UM_BIDIRECTIONAL=False``, no
functional links and both neighbors strictly inside the U3 gate. Their
circular mean has the midpoint lift. Single-target UM mixes its incident
oriented gaps; an all-target immutable stage applies cycle diffusion.

Separate C6 observers derive local default-bidirectional phase Jacobians
through the shared phasor-response owner. Their conditional joint domain
bounds nonlinear phase oscillation and reserves EPI margin for future
phase forcing under declared held nodal-Euler intervals. Tangent contraction
and the nonlinear oscillation bound remain distinct results.
Additive defect observers retain signed mean drift and finite reserve
telescopes. A separate conditional uniform tube requires independent error
and signed mean-prefix hypotheses; finite measured extrema do not supply them.
A finite opposite-pair projection closes a partial nonlinear phase/EPI
observable in the winding chamber, keeping common means and full-state
evolution separate from that reduced recurrence.

These detached algebraic observations do not infer a winding integer from
the supplied rational gap sum, admit an operator word, bind a live graph,
or certify a binary64 trajectory. See ``COUPLING_WINDING_PERSISTENCE.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from typing import TYPE_CHECKING

from .._exact_time import exact_or_represented_real as _rational
from ..constants.canonical import DELTA_PHI_MAX, UM_THETA_PUSH
from ._cycle_algebra import (
    Matrix, Vector, c6_pair_sums, dot, laplacian_matrix, ordered_vector,
)
from ._exact_linear_algebra import exact_square_matrix_power, exact_square_matrix_product

if TYPE_CHECKING:
    from .phase_response import PhaseResponseReference

__all__ = [
    "CouplingGapStep", "observe_coupling_gap_step",
    "C6WindingPhaseReference", "C6WindingPhaseObservation",
    "derive_c6_winding_phase_response", "observe_c6_winding_phase_response",
    "C6WindingJointDomain", "C6WindingJointStep",
    "derive_c6_winding_joint_domain", "observe_c6_winding_joint_domain",
    "C6WindingDefect", "C6WindingDefectPrefix", "C6WindingUniformDefectBound",
    "observe_c6_winding_defect", "bound_c6_winding_defect_prefix",
    "bound_c6_winding_uniform_defects",
    "C6WindingPairingReference", "C6WindingPairingObservation",
    "c6_centered_opposite_pairs", "derive_c6_winding_pairing",
    "observe_c6_winding_pairing",
]


def _gap_vector(gaps) -> Vector:
    values = ordered_vector(gaps, "gap")
    if len(values) < 3:
        raise ValueError("a simple cycle requires at least three gaps")
    return values


def _gap_matrix(count: int, eta: Fraction, target: int | None) -> Matrix:
    if target is None:
        return tuple(
            tuple(Fraction(i == j) - eta * value for j, value in enumerate(row))
            for i, row in enumerate(laplacian_matrix(count))
        )
    rows = [
        [Fraction(i == j) for j in range(count)] for i in range(count)
    ]
    half = eta / 2
    left, right = (target - 1) % count, target
    rows[left][left], rows[left][right] = 1 - half, half
    rows[right][left], rows[right][right] = half, 1 - half
    return tuple(tuple(row) for row in rows)


@dataclass(frozen=True)
class CouplingGapStep:
    """One exact interval-preserving gap step in a declared UM model.

    The spread is ``sum((gap-mean_gap)**2)/2``. No phase-loop closure or
    runtime provenance is inferred from these supplied gap coordinates.
    """

    input_gaps: Vector
    output_gaps: Vector
    eta: Fraction
    phase_gate: Fraction
    mode: str
    target: int | None
    matrix: Matrix
    gap_sum: Fraction
    mean_gap: Fraction
    spread_before: Fraction
    spread_after: Fraction
    spread_drop: Fraction
    expected_spread_drop: Fraction
    sum_residual: Fraction
    minimum_before: Fraction
    maximum_before: Fraction
    minimum_after: Fraction
    maximum_after: Fraction
    invariant_interval_preserved: bool


def observe_coupling_gap_step(
    gaps,
    *,
    target=None,
    eta=UM_THETA_PUSH,
    phase_gate=DELTA_PHI_MAX,
) -> CouplingGapStep:
    """Observe one exact target-only UM gap map, without graph mutation.

    ``target=None`` means one simultaneous all-target Jacobi stage. An
    integer target changes its two incident gaps at indices ``target-1``
    and ``target``. Both realizations require a fixed simple undirected
    cycle, target-only UM, no functional-link creation and two compatible
    neighbors per target when interpreted as canonical phase operations.

    Exact rational inputs are preserved. Other real inputs retain their
    shared materialized binary64 rational value; in particular the default
    eta is the represented canonical UM factor. Require ``0 < eta <= 1``
    and every gap strictly inside the declared gate, itself at most the
    represented canonical pi/2 limit. A rational sum does not certify an
    exact multiple of mathematical 2*pi. An external phase observation must
    establish cycle closure and winding separately.
    """
    values = _gap_vector(gaps)
    count = len(values)
    coefficient = _rational(eta, "eta")
    gate = _rational(phase_gate, "phase_gate")
    canonical_gate = Fraction.from_float(float(DELTA_PHI_MAX))
    if not 0 < coefficient <= 1:
        raise ValueError("eta must lie in (0, 1]")
    if not 0 < gate <= canonical_gate:
        raise ValueError("phase_gate must lie in (0, canonical pi/2]")
    if any(abs(value) >= gate for value in values):
        raise ValueError("every gap must lie strictly inside the phase gate")
    if target is not None:
        if isinstance(target, bool) or not isinstance(target, Integral):
            raise TypeError("target must be an integer cycle index or None")
        target = int(target)
        if not 0 <= target < count:
            raise ValueError("target must belong to the declared cycle")

    matrix = _gap_matrix(count, coefficient, target)
    output = tuple(
        sum((weight * value for weight, value in zip(row, values)), Fraction(0))
        for row in matrix
    )
    total = sum(values, Fraction(0))
    mean = total / count
    before = sum(((value - mean) ** 2 for value in values), Fraction(0)) / 2
    after = sum(((value - mean) ** 2 for value in output), Fraction(0)) / 2
    if target is None:
        edge_square = sum(
            ((values[(i + 1) % count] - values[i]) ** 2 for i in range(count)),
            Fraction(0),
        )
        two_hop_square = sum(
            (
                (values[(i + 1) % count] - values[(i - 1) % count]) ** 2
                for i in range(count)
            ),
            Fraction(0),
        )
        expected = (
            coefficient * (1 - coefficient) * edge_square / 2
            + coefficient**2 * two_hop_square / 8
        )
    else:
        difference = values[target] - values[(target - 1) % count]
        expected = coefficient * (2 - coefficient) * difference**2 / 4
    drop = before - after
    residual = sum(output, Fraction(0)) - total
    lower, upper = min(values), max(values)
    next_lower, next_upper = min(output), max(output)
    interval_preserved = lower <= next_lower <= next_upper <= upper
    if residual or drop != expected or drop < 0 or not interval_preserved:
        raise RuntimeError("exact Coupling gap transport lost its identities")
    return CouplingGapStep(
        input_gaps=values,
        output_gaps=output,
        eta=coefficient,
        phase_gate=gate,
        mode="all_targets" if target is None else "single_target",
        target=target,
        matrix=matrix,
        gap_sum=total,
        mean_gap=mean,
        spread_before=before,
        spread_after=after,
        spread_drop=drop,
        expected_spread_drop=expected,
        sum_residual=residual,
        minimum_before=lower,
        maximum_before=upper,
        minimum_after=next_lower,
        maximum_after=next_upper,
        invariant_interval_preserved=interval_preserved,
    )


@dataclass(frozen=True)
class C6WindingPhaseReference:
    """Exact local UM/IL phase matrices around one winding of ordered C6.

    Rotation is neutral. The quotient bound concerns centered tangent
    energy, not the full nonlinear phase map or complete engine evolution.
    """

    coupling_phase_factor: Fraction
    coherence_phase_factor: Fraction
    coupling_response: PhaseResponseReference
    coherence_response: PhaseResponseReference
    coupling_matrix: Matrix
    coherence_matrix: Matrix
    product_matrix: Matrix
    modal_multipliers: Vector
    quotient_energy_bound: Fraction
    rotation_identity_residuals: Vector
    strict_quotient_contraction: bool


def derive_c6_winding_phase_response(
    *, coupling_phase_factor, coherence_phase_factor,
) -> C6WindingPhaseReference:
    """Derive exact C6 Jacobians through the shared phasor-response kernel.

    The declared mathematical base is theta_i=i*pi/3 modulo 2*pi, in node
    order 0..5 on the simple cycle. Adjacent phases are strictly inside the
    canonical pi/2 gate, while every nonedge is outside. This identifies
    neither a live graph nor a represented phase vector. Bidirectional UM
    uses every closed cycle neighborhood and all six targets. Every receiver
    gets three source proposals whose base displacements +t*pi/3,0,-t*pi/3
    cancel after averaging. They do not vanish separately.

    If B is cycle adjacency, the exact mean responses are R_IL=B/2 and
    R_closed=I/2+B/4. The receiver response is W=(I+B)*R_closed/3. Thus
    U=(1-t)*I+t*W, M=(1-alpha)*I+alpha*B/2 and P=M*U. The Fourier
    multipliers, in k=0..5 order, are (1,l1,l2,l3,l2,l1), where
    l1=(1-t/2)*(1-alpha/2), l2=(1-t)*(1-3*alpha/2), and
    l3=(1-t)*(1-2*alpha). Symmetry gives the optimal centered quadratic
    bound q=max(l1**2,l2**2,l3**2). The constant rotation mode remains one.

    Factors lie in [0,1]; t=0 is a detached control rather than admitted UM.
    Rational source coefficients are exact; other supported reals retain
    their binary64 rational values. The exact cosine Gram avoids replacing
    mathematical pi by math.pi. No phase gate, finite neighborhood, winding
    preparation, binary64 derivative or complete operator word is certified.
    """
    from .phase_response import derive_phase_response

    t = _rational(coupling_phase_factor, "coupling_phase_factor")
    alpha = _rational(coherence_phase_factor, "coherence_phase_factor")
    if not 0 <= t <= 1 or not 0 <= alpha <= 1:
        raise ValueError("phase factors must lie in [0,1]")
    first_row = (Fraction(1), Fraction(1, 2), Fraction(-1, 2),
                 Fraction(-1), Fraction(-1, 2), Fraction(1, 2))
    gram = tuple(tuple(first_row[(i - j) % 6] for j in range(6)) for i in range(6))
    neighbors = tuple(((i - 1) % 6, (i + 1) % 6) for i in range(6))
    closed = tuple((i, *row) for i, row in enumerate(neighbors))
    coupling = derive_phase_response(
        cosine_gram=gram, mean_neighbors=closed, receiver_sources=closed, phase_factor=t,
    )
    coherence = derive_phase_response(
        cosine_gram=gram, mean_neighbors=neighbors,
        receiver_sources=tuple((i,) for i in range(6)), phase_factor=alpha,
    )
    product = exact_square_matrix_product(coherence.jacobian, coupling.jacobian)
    l1 = (1 - t / 2) * (1 - alpha / 2)
    l2 = (1 - t) * (1 - 3 * alpha / 2)
    l3 = (1 - t) * (1 - 2 * alpha)
    multipliers = (Fraction(1), l1, l2, l3, l2, l1)
    bound = max(value**2 for value in multipliers[1:])
    rotation = tuple(sum(row, Fraction(0)) - 1 for row in product)
    if (any(rotation) or not 0 <= bound <= 1
            or any(product[i][j] != product[j][i] for i in range(6) for j in range(6))):
        raise RuntimeError("exact C6 phase response lost its quotient or rotation identities")
    return C6WindingPhaseReference(
        t, alpha, coupling, coherence, coupling.jacobian, coherence.jacobian,
        product, multipliers, bound, rotation, bound < 1,
    )


@dataclass(frozen=True)
class C6WindingPhaseObservation:
    """One exact tangent action with rotation removed only from its energy.

    The phase-pressure vector is multiplied by mathematical pi, so it stays
    rational. It excludes the pressure channel weight and nodal capacity.
    """

    reference: C6WindingPhaseReference
    direction: Vector
    mean: Fraction
    centered_direction: Vector
    after_coupling: Vector
    after_coherence: Vector
    energy_before: Fraction
    energy_after: Fraction
    energy_change: Fraction
    energy_bound_residual: Fraction
    mean_identity_residual: Fraction
    phase_pressure_pi_scaled: Vector


def observe_c6_winding_phase_response(
    reference, *, direction,
) -> C6WindingPhaseObservation:
    """Apply the rebuilt C6 reference to six ordered tangent coordinates.

    Matrices act on the original direction. Half squared norms are computed
    after subtracting its arithmetic mean. energy_bound_residual is
    E_after-q*E_before, hence nonpositive. This bound need not be attained by
    every direction. The phase-pressure tangent is (B/2-I)*postphase after
    multiplication by mathematical pi; no floating approximation of pi is
    inserted. Public matrices and other reference caches are always rebuilt.
    """
    if not isinstance(reference, C6WindingPhaseReference):
        raise TypeError("reference must be a C6WindingPhaseReference")
    reference = derive_c6_winding_phase_response(
        coupling_phase_factor=reference.coupling_phase_factor,
        coherence_phase_factor=reference.coherence_phase_factor,
    )
    values = ordered_vector(direction, "phase tangent")
    if len(values) != 6:
        raise ValueError("C6 phase tangent requires exactly six ordered coordinates")
    mean = sum(values, Fraction(0)) / 6
    centered = tuple(value - mean for value in values)
    coupled = tuple(dot(row, values) for row in reference.coupling_matrix)
    after = tuple(dot(row, coupled) for row in reference.coherence_matrix)
    centered_after = tuple(value - mean for value in after)
    before_energy = dot(centered, centered) / 2
    after_energy = dot(centered_after, centered_after) / 2
    residual = after_energy - reference.quotient_energy_bound * before_energy
    mean_residual = sum(after, Fraction(0)) / 6 - mean
    pressure = tuple(-dot(row, after) for row in laplacian_matrix(6))
    if (mean_residual or residual > 0
            or after != tuple(dot(row, values) for row in reference.product_matrix)):
        raise RuntimeError("exact C6 tangent action lost its centered energy or mean identity")
    return C6WindingPhaseObservation(
        reference, values, mean, centered, coupled, after, before_energy,
        after_energy, after_energy - before_energy, residual, mean_residual, pressure,
    )


@dataclass(frozen=True)
class C6WindingJointDomain:
    """Conditional exact phase-contraction and held nodal-Euler domain.

    The phase reference supplies the common declared geometry and factors.
    Its tangent energy bound is not used as a nonlinear contraction bound.
    """

    phase_reference: C6WindingPhaseReference
    coupling_phase_factor: Fraction
    coherence_phase_factor: Fraction
    capacity: Fraction
    epi_weight: Fraction
    phase_weight: Fraction
    timestep: Fraction
    epi_lower: Fraction
    epi_upper: Fraction
    max_phase_oscillation_pi: Fraction
    closed_mean_partial_lower: Fraction
    nonlinear_oscillation_factor: Fraction
    forcing_step_factor: Fraction
    epi_phase_budget_weight: Fraction
    nodal_euler_matrix: Matrix
    epi_disagreement_factor: Fraction
    strict_epi_convergence: bool


def derive_c6_winding_joint_domain(
    *, coupling_phase_factor, coherence_phase_factor, capacity, epi_weight,
    phase_weight, timestep, epi_lower, epi_upper,
) -> C6WindingJointDomain:
    """Derive an exact nonlinear phase bound and positive EPI reserve.

    Scope is the prepared simple unit C6 with uniform held positive capacity,
    all-target bidirectional UM followed by simultaneous IL, and the regular
    winding chart of the phase reference. Let z be phase errors divided by
    mathematical pi. The common phase interval has width at most 1/12.

    In this invariant phase box, monotonicity places each closed phasor mean
    error in the same interval. Every contributing phase then differs from
    its mean by at most 5*pi/12, so its cosine exceeds 1/4. The resultant
    magnitude is below 5/2, hence each included mean partial exceeds 1/10.
    UM's three-source receiver average therefore has each supported radius-
    two partial at least t/30. Any two C6 rows share at least four such
    columns, yielding the nonlinear oscillation factor rho=1-2*t/15.
    IL is an exact convex midpoint map in this box and cannot increase
    oscillation. This bound is derived independently of tangent eigenvalues.

    For L=I-Adj/2, canonical phase pressure is -L*z. Constant capacity makes
    its capacity-gradient channel zero, and regular fixed support makes its
    topology-gradient channel zero. Write s=h*nu*w_epi and b=h*nu*w_phase.
    The refreshed, then held exact Euler interval is
    x_next=(I-s*L)*x-b*L*z_next. Require s<=1 so its EPI matrix is stochastic.
    If d=osc(z), B=b*rho/(1-rho), then min(x)-B*d is nondecreasing and
    max(x)+B*d is nonincreasing under admitted conditional steps. Requiring
    these reserves inside [epi_lower,epi_upper] gives a positive unclipped
    band through every affine held-input substep, not only at the endpoint.

    With 0<s<1, the centered Euler factor is strictly below one. Together
    with geometrically decaying phase forcing this implies exact-model EPI
    convergence to its initial arithmetic mean. At s=0 or s=1 that general
    conclusion is unavailable; s=1 retains an alternating mode of modulus
    one. No localized EPI limit is claimed from persistent winding alone.

    All coefficients are exact or represented rational inputs. This does
    not admit an operator word, infer a live phase chart, authenticate phase
    endpoints or certify binary64 dynamics. Application EPI/capacity gates
    must still be checked. Lower/upper bounds are a declared positive
    subinterval of the canonical unit EPI range, not inferred defaults.
    """
    phase_reference = derive_c6_winding_phase_response(
        coupling_phase_factor=coupling_phase_factor,
        coherence_phase_factor=coherence_phase_factor,
    )
    t, alpha = phase_reference.coupling_phase_factor, phase_reference.coherence_phase_factor
    nu = _rational(capacity, "capacity")
    e = _rational(epi_weight, "epi_weight")
    p = _rational(phase_weight, "phase_weight")
    h = _rational(timestep, "timestep")
    lower = _rational(epi_lower, "epi_lower")
    upper = _rational(epi_upper, "epi_upper")
    if t <= 0:
        raise ValueError("joint phase budget requires a strictly positive coupling factor")
    if nu <= 0 or e <= 0 or p < 0 or h < 0:
        raise ValueError("positive capacity/EPI weight and nonnegative phase weight/time are required")
    if not 0 < lower <= upper <= 1:
        raise ValueError("positive EPI band must satisfy 0 < lower <= upper <= 1")
    s = h * nu * e
    if s > 1:
        raise ValueError("held Euler requires timestep*capacity*epi_weight <= 1")
    rho = 1 - 2 * t / 15
    b = h * nu * p
    budget = b * rho / (1 - rho)
    laplacian = laplacian_matrix(6)
    matrix = tuple(tuple(Fraction(i == j) - s * value for j, value in enumerate(row))
                   for i, row in enumerate(laplacian))
    disagreement = max(abs(1 - s / 2), abs(1 - 3 * s / 2), abs(1 - 2 * s))
    if (not 0 <= rho < 1 or any(value < 0 for row in matrix for value in row)
            or any(sum(row, Fraction(0)) != 1 for row in matrix)):
        raise RuntimeError("exact C6 joint domain lost a stochastic or contraction identity")
    return C6WindingJointDomain(
        phase_reference, t, alpha, nu, e, p, h, lower, upper, Fraction(1, 12),
        Fraction(1, 10), rho, b, budget, matrix, disagreement, disagreement < 1,
    )


@dataclass(frozen=True)
class C6WindingJointStep:
    """One supplied phase pair and its conditional exact nodal interval.

    Passing the interval and oscillation inequalities does not establish
    that the supplied pair was produced by UM/IL, even in exact arithmetic.
    There is no execution seal or inference of future runtime behavior.
    """

    reference: C6WindingJointDomain
    phase_before_pi: Vector
    phase_after_pi: Vector
    phase_oscillation_before: Fraction
    phase_oscillation_after: Fraction
    phase_contraction_slack: Fraction
    phase_interval_nesting_slack: Vector
    epi_before: Vector
    epi_after: Vector
    phase_pressure: Vector
    modeled_pressure: Vector
    modeled_rate: Vector
    lower_reserve_before: Fraction
    lower_reserve_after: Fraction
    upper_reserve_before: Fraction
    upper_reserve_after: Fraction
    lower_reserve_gain: Fraction
    upper_reserve_gain: Fraction
    mean_before: Fraction
    mean_after: Fraction
    mean_identity_residual: Fraction


def _rebuild_c6_winding_joint_domain(reference):
    if not isinstance(reference, C6WindingJointDomain):
        raise TypeError("reference must be a C6WindingJointDomain")
    return derive_c6_winding_joint_domain(
        coupling_phase_factor=reference.coupling_phase_factor,
        coherence_phase_factor=reference.coherence_phase_factor,
        capacity=reference.capacity, epi_weight=reference.epi_weight,
        phase_weight=reference.phase_weight, timestep=reference.timestep,
        epi_lower=reference.epi_lower, epi_upper=reference.epi_upper,
    )


def _c6_joint_vector(values, label):
    result = ordered_vector(values, label)
    if len(result) != 6:
        raise ValueError("C6 joint observation requires six coordinates per vector")
    return result


def _c6_joint_nodal_model(reference, phase, epi):
    laplacian = laplacian_matrix(6)
    phase_pressure = tuple(-dot(row, phase) for row in laplacian)
    epi_pressure = tuple(-reference.epi_weight * dot(row, epi) for row in laplacian)
    pressure = tuple(left + reference.phase_weight * right
                     for left, right in zip(epi_pressure, phase_pressure, strict=True))
    rate = tuple(reference.capacity * value for value in pressure)
    final = tuple(dot(row, epi) + reference.forcing_step_factor * source
                  for row, source in zip(reference.nodal_euler_matrix, phase_pressure, strict=True))
    if final != tuple(value + reference.timestep * delta
                      for value, delta in zip(epi, rate, strict=True)):
        raise RuntimeError("exact C6 nodal interval lost its nodal rate identity")
    return phase_pressure, pressure, rate, final


def observe_c6_winding_joint_domain(
    reference, *, phase_before_pi, phase_after_pi, epi,
) -> C6WindingJointStep:
    """Check supplied phase-step inequalities and apply the exact nodal law.

    Before/after phase data are ordered errors divided by mathematical pi
    when interpreted as the declared exact winding chart. Their realized
    origin is not inferred. Normalizing captured floats by represented pi
    instead is an external model identification with a separate defect.
    The observer requires nested phase intervals and the nonlinear
    oscillation bound, as well as the initial EPI reserve. Every public
    reference cache is rebuilt before using any coefficient or matrix.
    """
    reference = _rebuild_c6_winding_joint_domain(reference)
    before = ordered_vector(phase_before_pi, "phase_before_pi")
    after = ordered_vector(phase_after_pi, "phase_after_pi")
    values = ordered_vector(epi, "epi")
    if any(len(vector) != 6 for vector in (before, after, values)):
        raise ValueError("C6 joint observation requires six coordinates per vector")
    d_before, d_after = max(before) - min(before), max(after) - min(after)
    if d_before > reference.max_phase_oscillation_pi:
        raise ValueError("phase oscillation exceeds the declared winding box")
    nesting = (min(after) - min(before), max(before) - max(after))
    contraction = reference.nonlinear_oscillation_factor * d_before - d_after
    if any(value < 0 for value in nesting) or contraction < 0:
        raise ValueError("supplied phase endpoints fail nesting or nonlinear contraction")
    budget = reference.epi_phase_budget_weight
    lower_before = min(values) - budget * d_before
    upper_before = max(values) + budget * d_before
    if lower_before < reference.epi_lower or upper_before > reference.epi_upper:
        raise ValueError("initial EPI field lacks the declared future phase-forcing reserve")
    phase_pressure, pressure, rate, final = _c6_joint_nodal_model(reference, after, values)
    lower_after = min(final) - budget * d_after
    upper_after = max(final) + budget * d_after
    mean_before = sum(values, Fraction(0)) / 6
    mean_after = sum(final, Fraction(0)) / 6
    if lower_after < lower_before or upper_after > upper_before or mean_after != mean_before:
        raise RuntimeError("exact C6 nodal interval lost its reserve or mean identity")
    return C6WindingJointStep(
        reference, before, after, d_before, d_after, contraction, nesting, values,
        final, phase_pressure, pressure, rate, lower_before, lower_after,
        upper_before, upper_after, lower_after - lower_before, upper_before - upper_after,
        mean_before, mean_after, mean_after - mean_before,
    )


def _oscillation(values):
    return max(values) - min(values)


@dataclass(frozen=True)
class C6WindingDefect:
    """Exact additive defects of supplied endpoints, without a causal seal.

    Endpoint data may fail the earlier defect-free phase or reserve domain.
    This observation retains those failures instead of assuming them away.
    """

    reference: C6WindingJointDomain
    phase_before_pi: Vector
    phase_after_pi: Vector
    epi_before: Vector
    epi_after: Vector
    phase_oscillation_before: Fraction
    phase_oscillation_after: Fraction
    phase_oscillation_defect: Fraction
    phase_interval_expansion: Fraction
    phase_pressure: Vector
    modeled_pressure: Vector
    modeled_rate: Vector
    modeled_epi_after: Vector
    endpoint_defect: Vector
    mean_defect: Fraction
    centered_defect: Vector
    centered_defect_oscillation: Fraction
    endpoint_defect_infinity: Fraction
    mean_before: Fraction
    mean_after: Fraction
    mean_identity_residual: Fraction
    lower_reserve_before: Fraction
    lower_reserve_after: Fraction
    upper_reserve_before: Fraction
    upper_reserve_after: Fraction
    lower_loss_bound: Fraction
    upper_loss_bound: Fraction
    lower_bound_slack: Fraction
    upper_bound_slack: Fraction
    absolute_defect_cost: Fraction
    phase_box_before: bool
    phase_box_after: bool


def observe_c6_winding_defect(
    reference, *, phase_before_pi, phase_after_pi, epi_before, epi_after,
) -> C6WindingDefect:
    """Measure additive phase/EPI defects against the rebuilt exact owner.

    Set v=osc(z), epsilon=max(0,v_next-rho*v), and
    delta=x_next-[T*x-b*L*z_next]. The mean identity is exactly
    mean(x_next)-mean(x)=mean(delta). Uniform per-step mean error alone
    cannot bound its signed accumulated drift.

    For the original reserves ell=min(x)-B*v, u=max(x)+B*v, define
    C=B+b. Then ell_next>=ell-[C*epsilon-min(delta)] and
    u_next<=u+[C*epsilon+max(delta)]. These loss bounds are signed and
    retain common-drift cancellation. Both are at most the conservative
    absolute cost C*epsilon+norm_inf(delta). The centered defect and mean
    are reported separately. No uniform future error bound is inferred.

    Phase vectors use the earlier declared pi-scaled chart convention.
    Their inequalities alone do not authenticate a nonlinear phase map,
    live support, operator admission, or a binary64 error class.
    """
    reference = _rebuild_c6_winding_joint_domain(reference)
    z0 = _c6_joint_vector(phase_before_pi, "phase_before_pi")
    z1 = _c6_joint_vector(phase_after_pi, "phase_after_pi")
    x0 = _c6_joint_vector(epi_before, "epi_before")
    x1 = _c6_joint_vector(epi_after, "epi_after")
    v0, v1 = _oscillation(z0), _oscillation(z1)
    epsilon = max(Fraction(0), v1 - reference.nonlinear_oscillation_factor * v0)
    expansion = max(Fraction(0), min(z0) - min(z1), max(z1) - max(z0))
    phase, pressure, rate, model = _c6_joint_nodal_model(reference, z1, x0)
    delta = tuple(actual - predicted for actual, predicted in zip(x1, model, strict=True))
    mu = sum(delta, Fraction(0)) / 6
    centered = tuple(value - mu for value in delta)
    delta_infinity = max(abs(value) for value in delta)
    mean0, mean1 = sum(x0, Fraction(0)) / 6, sum(x1, Fraction(0)) / 6
    mean_residual = mean1 - mean0 - mu
    budget = reference.epi_phase_budget_weight
    low0, low1 = min(x0) - budget * v0, min(x1) - budget * v1
    high0, high1 = max(x0) + budget * v0, max(x1) + budget * v1
    phase_cost = (budget + reference.forcing_step_factor) * epsilon
    lower_loss, upper_loss = phase_cost - min(delta), phase_cost + max(delta)
    lower_slack = low1 - low0 + lower_loss
    upper_slack = high0 + upper_loss - high1
    if mean_residual or lower_slack < 0 or upper_slack < 0:
        raise RuntimeError("exact C6 additive defect lost its mean or reserve inequality")
    return C6WindingDefect(
        reference, z0, z1, x0, x1, v0, v1, epsilon, expansion,
        phase, pressure, rate, model, delta, mu, centered, _oscillation(centered),
        delta_infinity, mean0, mean1, mean_residual, low0, low1, high0, high1,
        lower_loss, upper_loss, lower_slack, upper_slack, phase_cost + delta_infinity,
        v0 <= reference.max_phase_oscillation_pi, v1 <= reference.max_phase_oscillation_pi,
    )


@dataclass(frozen=True)
class C6WindingDefectPrefix:
    """Exact finite telescope of adjacent supplied observations.

    Matching endpoints establish algebraic adjacency, not runtime causality.
    Finite error extrema or mean-prefix bounds do not become future bounds.
    """

    reference: C6WindingJointDomain
    observations: tuple[C6WindingDefect, ...]
    phase_oscillation_envelope: Vector
    mean_defect_prefix: Vector
    signed_lower_loss_prefix: Vector
    signed_upper_loss_prefix: Vector
    absolute_defect_cost_prefix: Vector
    lower_reserve_bounds: Vector
    upper_reserve_bounds: Vector
    mean_prefix_bound: Fraction
    phase_class_preserved: bool
    epi_reserve_preserved: bool


def bound_c6_winding_defect_prefix(reference, *, observations) -> C6WindingDefectPrefix:
    """Rebuild a nonempty finite endpoint chain and telescope signed losses.

    The returned sufficient phase/reserve flags apply only to this supplied
    finite chain. No independent uniform-error premise or future behavior
    is inferred from it. All cached observations and references are rebuilt.
    """
    reference = _rebuild_c6_winding_joint_domain(reference)
    if isinstance(observations, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("observations must be an ordered sequence")
    items = tuple(observations)
    if not items:
        raise ValueError("a finite defect prefix requires at least one observation")
    rebuilt = []
    for item in items:
        if not isinstance(item, C6WindingDefect):
            raise TypeError("every observation must be a C6WindingDefect")
        if _rebuild_c6_winding_joint_domain(item.reference) != reference:
            raise ValueError("all observations must share the same joint-domain coefficients")
        step = observe_c6_winding_defect(
            reference, phase_before_pi=item.phase_before_pi, phase_after_pi=item.phase_after_pi,
            epi_before=item.epi_before, epi_after=item.epi_after,
        )
        if rebuilt and (rebuilt[-1].epi_after != step.epi_before
                        or rebuilt[-1].phase_after_pi != step.phase_before_pi):
            raise ValueError("finite defect observations must have exactly adjacent endpoints")
        rebuilt.append(step)
    first = rebuilt[0]
    phase = [first.phase_oscillation_before]
    means, lower_losses, upper_losses, absolute_costs = ([Fraction(0)] for _ in range(4))
    lower_bounds, upper_bounds = [first.lower_reserve_before], [first.upper_reserve_before]
    for step in rebuilt:
        phase.append(reference.nonlinear_oscillation_factor * phase[-1] + step.phase_oscillation_defect)
        means.append(means[-1] + step.mean_defect)
        lower_losses.append(lower_losses[-1] + step.lower_loss_bound)
        upper_losses.append(upper_losses[-1] + step.upper_loss_bound)
        absolute_costs.append(absolute_costs[-1] + step.absolute_defect_cost)
        lower_bounds.append(first.lower_reserve_before - lower_losses[-1])
        upper_bounds.append(first.upper_reserve_before + upper_losses[-1])
        if (step.phase_oscillation_after > phase[-1]
                or step.mean_after != first.mean_before + means[-1]
                or step.lower_reserve_after < lower_bounds[-1]
                or step.upper_reserve_after > upper_bounds[-1]):
            raise RuntimeError("exact C6 finite defect telescope lost a prefix identity")
    return C6WindingDefectPrefix(
        reference, tuple(rebuilt), tuple(phase), tuple(means), tuple(lower_losses),
        tuple(upper_losses), tuple(absolute_costs), tuple(lower_bounds), tuple(upper_bounds),
        max(abs(value) for value in means),
        max(phase) <= reference.max_phase_oscillation_pi,
        min(lower_bounds) >= reference.epi_lower and max(upper_bounds) <= reference.epi_upper,
    )


@dataclass(frozen=True)
class C6WindingUniformDefectBound:
    """Conditional invariant envelope under separately declared uniform errors.

    The mean-prefix allowance is an independent signed-prefix hypothesis.
    The strict contraction flag concerns homogeneous Euler diffusion only;
    persistent additive defects can sustain nonzero disagreement.
    """

    reference: C6WindingJointDomain
    initial_epi: Vector
    initial_phase_oscillation: Fraction
    initial_mean: Fraction
    initial_range_functional: Fraction
    phase_defect_bound: Fraction
    centered_epi_defect_bound: Fraction
    mean_prefix_bound: Fraction
    phase_bound: Fraction
    forcing_oscillation_bound: Fraction
    diffusion_two_step_matrix: Matrix
    diffusion_two_step_factor: Fraction
    range_functional_factor: Fraction
    range_functional_bound: Fraction
    epi_envelope_lower: Fraction
    epi_envelope_upper: Fraction
    phase_class_preserved: bool
    epi_band_preserved: bool
    joint_domain_preserved: bool
    strict_epi_range_contraction: bool


def bound_c6_winding_uniform_defects(
    reference, *, initial_epi, initial_phase_oscillation, phase_defect_bound,
    centered_epi_defect_bound, mean_prefix_bound,
) -> C6WindingUniformDefectBound:
    """Derive a conditional phase/range tube without square-root estimates.

    Assume epsilon_n<=epsilon, osc(delta_n)<=E, and independently
    abs(sum_{j<n}mean(delta_j))<=M for every considered prefix. E equals
    osc(delta_centered), not its infinity norm. Uniform per-step mean error
    does not imply the signed-prefix hypothesis. Supplied finite observed
    maxima do not establish any of these assumptions for future execution.

    The phase envelope is D=max(v0,epsilon/(1-rho)). The complete additive
    nodal forcing has oscillation at most F=2*b*D+E. Compute the optimal
    oscillation factor sigma of T^2 from exact row overlap. On C6 it equals
    1-min(s^2,4*s*(1-s)). Define V(x)=osc(x)+osc(T*x); then
    V(T*x)<=r*V(x), r=(1+sigma)/2, and V(x_next)<=r*V(x)+2*F.
    Thus V<=R=max(V0,2*F/(1-r)) is a forward-invariant bound. The separate
    mean hypothesis yields x_i in [m0-M-5*R/6,m0+M+5*R/6]. The factor 5/6
    is the sharp six-coordinate mean/range inequality.

    If s=0 or s=1, sigma=1: nonzero F has no bound of this form and is
    rejected. Zero F retains R=V0 without a strict contraction claim.
    Failed phase/band conditions are returned as false sufficient-domain
    flags. No live chart, future schedule, clipping policy or error class
    is certified. The earlier tangent phase factor is never substituted
    for the nonlinear oscillation bound.
    """
    reference = _rebuild_c6_winding_joint_domain(reference)
    epi = _c6_joint_vector(initial_epi, "initial_epi")
    v0 = _rational(initial_phase_oscillation, "initial_phase_oscillation")
    epsilon = _rational(phase_defect_bound, "phase_defect_bound")
    error = _rational(centered_epi_defect_bound, "centered_epi_defect_bound")
    mean_bound = _rational(mean_prefix_bound, "mean_prefix_bound")
    if min(v0, epsilon, error, mean_bound) < 0:
        raise ValueError("phase, centered-error and mean-prefix bounds must be nonnegative")
    phase_bound = max(v0, epsilon / (1 - reference.nonlinear_oscillation_factor))
    forcing = 2 * reference.forcing_step_factor * phase_bound + error
    matrix = reference.nodal_euler_matrix
    squared = exact_square_matrix_power(matrix, 2)
    overlap = min(sum((min(a, b) for a, b in zip(left, right, strict=True)), Fraction(0))
                  for left in squared for right in squared)
    sigma = 1 - overlap
    s = reference.timestep * reference.capacity * reference.epi_weight
    if sigma != 1 - min(s**2, 4 * s * (1 - s)):
        raise RuntimeError("exact C6 two-step overlap lost its optimal coefficient")
    r = (1 + sigma) / 2
    v_epi = _oscillation(epi) + _oscillation(tuple(dot(row, epi) for row in matrix))
    if r == 1:
        if forcing:
            raise ValueError("nonzero uniform forcing requires strict two-step Euler contraction")
        bound = v_epi
    else:
        bound = max(v_epi, 2 * forcing / (1 - r))
    mean = sum(epi, Fraction(0)) / 6
    lower = mean - mean_bound - 5 * bound / 6
    upper = mean + mean_bound + 5 * bound / 6
    phase_ok = phase_bound <= reference.max_phase_oscillation_pi
    epi_ok = reference.epi_lower <= lower and upper <= reference.epi_upper
    return C6WindingUniformDefectBound(
        reference, epi, v0, mean, v_epi, epsilon, error, mean_bound, phase_bound,
        forcing, squared, sigma, r, bound, lower, upper, phase_ok, epi_ok,
        phase_ok and epi_ok, sigma < 1,
    )


def c6_centered_opposite_pairs(values) -> Vector:
    """Read three opposite-pair sums after removing the common rotation.

    Six ordered exact or represented real coordinates are required. This
    projection does not identify phases, a graph, or any realized symmetry.
    """
    values = _c6_joint_vector(values, "opposite-pair coordinate")
    twice_mean = sum(values, Fraction(0)) / 3
    return tuple(value - twice_mean for value in c6_pair_sums(values))


@dataclass(frozen=True)
class C6WindingPairingReference:
    """Exact finite phase/EPI quotient of the declared C6 phase chamber.

    The two coordinates are centered phase-pair and EPI-pair sums. Their
    closed evolution does not close the six-node state, preserve a phase
    mean, or transfer the full tangent energy bound to a nonlinear map.
    """

    reference: C6WindingJointDomain
    projection_matrix: Matrix
    receiver_average_matrix: Matrix
    coherence_matrix: Matrix
    phase_pair_factor: Fraction
    epi_pair_factor: Fraction
    postphase_to_epi_pair_factor: Fraction
    quotient_matrix: Matrix
    receiver_projection_residual: Matrix
    coherence_projection_residual: Matrix
    laplacian_projection_residual: Matrix
    quotient_spectral_radius: Fraction
    strict_quotient_decay: bool


def derive_c6_winding_pairing(reference) -> C6WindingPairingReference:
    """Derive the nonlinear opposite-pair quotient from the existing owner.

    Let Pz=(z_i+z_(i+3)-2*mean(z)) for i=0,1,2, K=(I+Adj)/3 and
    M=(1-alpha)I+alpha*Adj/2. In the established winding chamber the exact
    all-target UM phase update is (1-t)z+t*K*c, where c contains its closed
    phasor-mean errors. Opposite receiver triples cover all six sources,
    so P*K=0. Also P*M=(1-3*alpha/2)P and P*L=(3/2)P.

    Thus phase-pair sums obey h_phi_next=lambda2*h_phi for every c, with
    lambda2=(1-t)*(1-3*alpha/2). This reuses the earlier modal coefficient
    but proves its action on this finite nonlinear observable separately.
    After the shared nodal interval,
    h_x_next=(1-3*s/2)h_x-(3*b/2)h_phi_next,
    where s=h*nu*w_epi and b=h*nu*w_phase. The displayed 2x2 matrix acts
    on (h_phi,h_x), separately for each of the three pairs.

    For common fixed coefficients and repeated steps in this same chamber,
    positive duration gives a quotient spectral radius below one and hence
    decay of these pair defects. This is not a one-step norm contraction
    bound. At h=0 the EPI-pair coordinate is unchanged. At s=1 the quotient
    still decays, while an opposite-antisymmetric full EPI mode can persist.
    No complete-state convergence, future admission or binary64 symmetry
    follows. Public joint-reference caches are rebuilt before derivation.
    """
    reference = _rebuild_c6_winding_joint_domain(reference)
    laplacian = laplacian_matrix(6)
    receiver = tuple(tuple(Fraction(i == j) - 2 * value / 3 for j, value in enumerate(row))
                     for i, row in enumerate(laplacian))
    projection = tuple(tuple(Fraction(j in (i, i + 3)) - Fraction(1, 3) for j in range(6))
                       for i in range(3))
    coherence = reference.phase_reference.coherence_matrix
    projected = tuple(tuple(tuple(dot(row, column) for column in zip(*matrix, strict=True))
                            for row in projection) for matrix in (receiver, coherence, laplacian))
    il_factor = 1 - 3 * reference.coherence_phase_factor / 2
    receiver_residual = projected[0]
    coherence_residual = tuple(
        tuple(value - il_factor * expected for value, expected in zip(row, target, strict=True))
        for row, target in zip(projected[1], projection, strict=True)
    )
    laplacian_residual = tuple(
        tuple(value - 3 * expected / 2 for value, expected in zip(row, target, strict=True))
        for row, target in zip(projected[2], projection, strict=True)
    )
    phase_factor = reference.phase_reference.modal_multipliers[2]
    epi_factor = 1 - 3 * reference.timestep * reference.capacity * reference.epi_weight / 2
    cross_factor = -3 * reference.forcing_step_factor / 2
    quotient = ((phase_factor, Fraction(0)), (cross_factor * phase_factor, epi_factor))
    spectral_radius = max(abs(phase_factor), abs(epi_factor))
    if (phase_factor != (1 - reference.coupling_phase_factor) * il_factor
            or any(value for matrix in (receiver_residual, coherence_residual, laplacian_residual)
                   for row in matrix for value in row)):
        raise RuntimeError("exact C6 pairing lost its receiver or Laplacian identity")
    return C6WindingPairingReference(
        reference, projection, receiver, coherence, phase_factor, epi_factor,
        cross_factor, quotient, receiver_residual, coherence_residual,
        laplacian_residual, spectral_radius, spectral_radius < 1,
    )


@dataclass(frozen=True)
class C6WindingPairingObservation:
    """One supplied closed-mean phase map and its exact nodal pair quotient.

    Closed means are constrained to the input phase interval but are not
    authenticated as phasor outputs. This necessary chamber condition alone
    does not establish the nonlinear rho bound or repeated reserve validity.
    """

    reference: C6WindingPairingReference
    phase_before_pi: Vector
    closed_mean_errors_pi: Vector
    phase_after_coupling_pi: Vector
    phase_after_coherence_pi: Vector
    phase_mean_before_pi: Fraction
    closed_mean_average_pi: Fraction
    phase_mean_after_coupling_pi: Fraction
    phase_mean_after_coherence_pi: Fraction
    phase_mean_identity_residual: Fraction
    phase_pair_before: Vector
    phase_pair_after_coupling: Vector
    phase_pair_after: Vector
    phase_pair_identity_residual: Vector
    epi_before: Vector
    epi_after: Vector
    epi_mean_before: Fraction
    epi_mean_after: Fraction
    epi_mean_identity_residual: Fraction
    epi_pair_before: Vector
    epi_pair_after: Vector
    epi_pair_identity_residual: Vector
    phase_pressure: Vector
    modeled_pressure: Vector
    modeled_rate: Vector
    phase_pressure_pairs: Vector
    total_pressure_pairs: Vector
    phase_pressure_pair_identity_residual: Vector
    total_pressure_pair_identity_residual: Vector
    phase_oscillation_before: Fraction
    phase_oscillation_after: Fraction
    phase_contraction_slack: Fraction
    meets_nonlinear_phase_contraction: bool
    lower_reserve_before: Fraction
    lower_reserve_after: Fraction
    upper_reserve_before: Fraction
    upper_reserve_after: Fraction
    epi_reserve_preserved: bool


def observe_c6_winding_pairing(
    reference, *, phase_before_pi, closed_mean_errors_pi, epi,
) -> C6WindingPairingObservation:
    """Evaluate supplied finite phase means and the exact nodal quotient.

    Accept either a C6WindingJointDomain or its derived pairing reference;
    every public cache is rebuilt from the joint primitive inputs. Six
    ordered z and c values must satisfy osc(z)<=1/12 and min(z)<=c_i<=max(z).
    Actual closed phasor means in the declared chamber obey that interval
    property, but arbitrary supplied values passing it need not be those
    means or satisfy the stronger nonlinear oscillation factor rho.

    Initial EPI must satisfy the existing B16 reserve. This ensures the
    single nodal endpoint stays in the positive source band: the supplied
    phase maps do not increase oscillation and B>=b on this factor domain.
    The output reserve and rho condition are reported separately; neither
    is silently assumed for the next call. Means remain separate from the
    centered pair readouts, and no source authentication or live execution
    evidence is created by this algebraic evaluation.
    """
    if isinstance(reference, C6WindingPairingReference):
        reference = reference.reference
    pairing = derive_c6_winding_pairing(reference)
    source = pairing.reference
    before = _c6_joint_vector(phase_before_pi, "phase_before_pi")
    closed = _c6_joint_vector(closed_mean_errors_pi, "closed_mean_errors_pi")
    values = _c6_joint_vector(epi, "epi")
    diameter = _oscillation(before)
    if diameter > source.max_phase_oscillation_pi:
        raise ValueError("phase oscillation exceeds the declared winding box")
    if min(closed) < min(before) or max(closed) > max(before):
        raise ValueError("supplied closed means must lie inside the initial phase interval")
    budget = source.epi_phase_budget_weight
    lower_before, upper_before = min(values) - budget * diameter, max(values) + budget * diameter
    if lower_before < source.epi_lower or upper_before > source.epi_upper:
        raise ValueError("initial EPI field lacks the declared future phase-forcing reserve")
    t = source.coupling_phase_factor
    coupled = tuple((1 - t) * value + t * dot(row, closed)
                    for value, row in zip(before, pairing.receiver_average_matrix, strict=True))
    after = tuple(dot(row, coupled) for row in pairing.coherence_matrix)
    final_diameter = _oscillation(after)
    phase, pressure, rate, final = _c6_joint_nodal_model(source, after, values)
    phase_before, phase_coupled, phase_after = (
        c6_centered_opposite_pairs(vector) for vector in (before, coupled, after)
    )
    epi_before, epi_after = (c6_centered_opposite_pairs(vector) for vector in (values, final))
    phase_pair_residual = tuple(
        actual - pairing.phase_pair_factor * previous
        for actual, previous in zip(phase_after, phase_before, strict=True)
    )
    epi_pair_residual = tuple(
        actual - pairing.epi_pair_factor * previous - pairing.postphase_to_epi_pair_factor * forcing
        for actual, previous, forcing in zip(epi_after, epi_before, phase_after, strict=True)
    )
    phase_pairs, pressure_pairs = c6_pair_sums(phase), c6_pair_sums(pressure)
    phase_pressure_residual = tuple(value + 3 * pair / 2 for value, pair in zip(phase_pairs, phase_after, strict=True))
    pressure_residual = tuple(
        value + 3 * (source.epi_weight * hx + source.phase_weight * hz) / 2
        for value, hx, hz in zip(pressure_pairs, epi_before, phase_after, strict=True)
    )
    mean_before, mean_closed, mean_coupled, mean_after, mean_epi, mean_final = (
        sum(vector, Fraction(0)) / 6 for vector in (before, closed, coupled, after, values, final)
    )
    mean_residual = mean_after - ((1 - t) * mean_before + t * mean_closed)
    lower_after, upper_after = min(final) - budget * final_diameter, max(final) + budget * final_diameter
    contraction = source.nonlinear_oscillation_factor * diameter - final_diameter
    if (any(phase_pair_residual + epi_pair_residual + phase_pressure_residual + pressure_residual)
            or mean_residual or mean_after != mean_coupled or mean_final != mean_epi
            or final_diameter > diameter
            or min(final) < source.epi_lower or max(final) > source.epi_upper
            or phase_coupled != tuple((1 - t) * value for value in phase_before)):
        raise RuntimeError("exact C6 finite pairing lost its quotient, mean or interval identity")
    return C6WindingPairingObservation(
        pairing, before, closed, coupled, after, mean_before, mean_closed, mean_coupled, mean_after,
        mean_residual, phase_before, phase_coupled, phase_after, phase_pair_residual, values, final,
        mean_epi, mean_final, mean_final - mean_epi, epi_before, epi_after, epi_pair_residual,
        phase, pressure, rate, phase_pairs, pressure_pairs, phase_pressure_residual, pressure_residual,
        diameter, final_diameter, contraction, contraction >= 0, lower_before, lower_after,
        upper_before, upper_after, lower_after >= source.epi_lower and upper_after <= source.epi_upper,
    )
