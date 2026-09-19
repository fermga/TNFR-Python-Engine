"""Exact P2 form/capacity feedback and a conditional binary64 capacity class.

On mutual-singleton equal-conductance support, synchronized zero phase and
zero topology-channel forcing give F=-w_vf*L_rw*nu. One target-only Coupling
capacity proposal followed by refreshed forced nodal Euler first gives a
detached rational model with a positive invariant EPI box. The separate
binary64 lattice theorem covers the represented capacity map on [1,2] with
fixed neighbor capacity one; its observer matches the shared production
arithmetic under declared IEEE assumptions. Neither result writes a graph
or certifies grammar admission, pressure refresh or future complete runtime.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

from .._exact_time import exact_or_represented_real
from ._cycle_algebra import Vector, ordered_vector

__all__ = [
    "P2CapacityFeedbackReference",
    "P2CapacityFeedbackCycle",
    "P2CapacityFeedbackBound",
    "derive_p2_capacity_feedback",
    "observe_p2_capacity_feedback_cycle",
    "bound_p2_capacity_feedback",
    "P2Binary64CouplingReference",
    "P2Binary64CouplingObservation",
    "derive_p2_binary64_coupling_lattice",
    "observe_p2_binary64_coupling_lattice",
]


@dataclass(frozen=True)
class P2CapacityFeedbackReference:
    """Exact coefficients and a proved box for this specific numeric model."""

    base_capacity: Fraction
    epi_weight: Fraction
    vf_weight: Fraction
    coupling_factor: Fraction
    timestep: Fraction
    max_capacity_gap: Fraction
    epi_lower: Fraction
    epi_upper: Fraction
    forcing_ratio: Fraction
    capacity_retention: Fraction
    step_coefficient: Fraction
    contraction_factor: Fraction
    strict_disagreement_contraction: bool


def derive_p2_capacity_feedback(
    *,
    base_capacity,
    epi_weight,
    vf_weight,
    coupling_factor,
    timestep,
    max_capacity_gap,
    epi_lower,
    epi_upper,
) -> P2CapacityFeedbackReference:
    """Validate an invariant box and a uniform lifted-disagreement envelope.

    Write c=base_capacity, k=w_vf/e, rho=1-gamma, s=h*e and A=max_capacity_gap.
    The box is 0<=a<=A, ell<=x0<=U-k*a, ell+k*a<=x1<=U. Its nonemptiness
    requires k*A<=U-ell. After Coupling a'=rho*a, and Euler in y=x+k*nu is
    row-stochastic provided s*(c+rho*A)<=1. Both event and flow preserve the box.

    The worst disagreement multiplier is
    beta=max(abs(1-2*s*c), abs(1-s*(2*c+rho*A))). beta<1 gives the conditional
    exact-model decay theorem. h=0 and the a=0 full-swap boundary remain valid
    noncontracting models. Hard clipping is unnecessary inside the declared
    interval; a runtime using another interval or soft clipping needs its own
    check. The positive lower bound is sufficient scalar-form headroom, not a
    complete grammar or application-precondition certificate.
    """
    inputs = (
        ("base_capacity", base_capacity),
        ("epi_weight", epi_weight),
        ("vf_weight", vf_weight),
        ("coupling_factor", coupling_factor),
        ("timestep", timestep),
        ("max_capacity_gap", max_capacity_gap),
        ("epi_lower", epi_lower),
        ("epi_upper", epi_upper),
    )
    c, e, w, gamma, h, limit, lower, upper = tuple(
        exact_or_represented_real(value, name) for name, value in inputs
    )
    if c <= 0 or e <= 0 or w < 0:
        raise ValueError(
            "base capacity and EPI weight must be positive; vf weight nonnegative"
        )
    if not 0 < gamma < 1:
        raise ValueError("coupling factor must be strictly between zero and one")
    if h < 0 or limit < 0:
        raise ValueError("timestep and maximum capacity gap must be nonnegative")
    if lower <= 0 or upper < lower:
        raise ValueError("the declared EPI interval must have 0 < lower <= upper")
    ratio, retention, step = w / e, 1 - gamma, h * e
    if ratio * limit > upper - lower:
        raise ValueError(
            "capacity shift must fit inside the declared invariant EPI box"
        )
    if step * (c + retention * limit) > 1:
        raise ValueError(
            "post-Coupling Euler coefficients must be convex on the whole box"
        )
    beta = max(abs(1 - 2 * step * c), abs(1 - step * (2 * c + retention * limit)))
    if beta > 1:
        raise RuntimeError("convex P2 coefficients lost their disagreement bound")
    return P2CapacityFeedbackReference(
        c,
        e,
        w,
        gamma,
        h,
        limit,
        lower,
        upper,
        ratio,
        retention,
        step,
        beta,
        beta < 1,
    )


def _reference(value):
    if type(value) is not P2CapacityFeedbackReference:
        raise TypeError("reference must be a P2CapacityFeedbackReference")
    return derive_p2_capacity_feedback(
        **{
            name: getattr(value, name)
            for name in (
                "base_capacity",
                "epi_weight",
                "vf_weight",
                "coupling_factor",
                "timestep",
                "max_capacity_gap",
                "epi_lower",
                "epi_upper",
            )
        }
    )


def _inputs(reference, capacity_gap, epi):
    ref = _reference(reference)
    gap = exact_or_represented_real(capacity_gap, "capacity_gap")
    x = ordered_vector(epi, "epi")
    if len(x) != 2:
        raise ValueError(
            "P2 feedback requires exactly two ordered scalar EPI coordinates"
        )
    if not 0 <= gap <= ref.max_capacity_gap:
        raise ValueError("capacity gap must lie in the declared [0, A] interval")
    return ref, gap, x


def _inside(ref, gap, epi):
    shift = ref.forcing_ratio * gap
    return (
        ref.epi_lower <= epi[0] <= ref.epi_upper - shift
        and ref.epi_lower + shift <= epi[1] <= ref.epi_upper
    )


def _metric_mean(c, gap, epi):
    return (c * epi[0] + (c + gap) * epi[1]) / (2 * c + gap)


@dataclass(frozen=True)
class P2CapacityFeedbackCycle:
    """One exact modeled cycle; domain flags do not authenticate a runtime."""

    reference: P2CapacityFeedbackReference
    capacity_gap: Fraction
    epi: Vector
    capacity_gap_after: Fraction
    capacity_after: Vector
    epi_after: Vector
    lifted_before: Vector
    lifted_after_event: Vector
    lifted_after: Vector
    modeled_pressure: Vector
    nodal_rate: Vector
    euler_mix: Vector
    flow_multiplier: Fraction
    disagreement_before: Fraction
    disagreement_after: Fraction
    lifted_disagreement_before: Fraction
    lifted_disagreement_after_event: Fraction
    lifted_disagreement_after: Fraction
    disagreement_identity_residual: Fraction
    lifted_identity_residual: Fraction
    arithmetic_mean_before: Fraction
    arithmetic_mean_after: Fraction
    arithmetic_mean_change: Fraction
    arithmetic_mean_flow_change: Fraction
    arithmetic_mean_identity_residual: Fraction
    metric_mean_before: Fraction
    metric_mean_after_event: Fraction
    metric_mean_after: Fraction
    metric_mean_change: Fraction
    metric_mean_reset_change: Fraction
    metric_mean_identity_residual: Fraction
    metric_flow_mean_identity_residual: Fraction
    before_invariant_domain: bool
    after_event_invariant_domain: bool
    after_invariant_domain: bool
    output_in_declared_interval: bool


def observe_p2_capacity_feedback_cycle(
    reference,
    *,
    capacity_gap,
    epi,
) -> P2CapacityFeedbackCycle:
    """Evaluate Coupling then Euler, retaining both mean-coordinate balances.

    For d=x0-x1 and q=d+k*a, a'=rho*a and q'=(1-s*(2*c+a'))*(q-k*gamma*a).
    The arithmetic mean changes only during flow, by -s*a'*(d+k*a')/2.
    The current H-mean changes only when the capacity metric resets, by
    c*gamma*a*d/((2*c+a)*(2*c+a')). This flow has zero H-weighted mean drift.

    EPI outside the invariant box is accepted as a detached counterexample and
    reported with false domain flags. The initial gap must still lie in [0,A].
    No clipping or ad hoc corrective pressure is applied to the modeled result.
    """
    ref, gap, x = _inputs(reference, capacity_gap, epi)
    c, k, s = ref.base_capacity, ref.forcing_ratio, ref.step_coefficient
    next_gap = ref.capacity_retention * gap
    capacity = (c + next_gap, c)
    difference = x[0] - x[1]
    q = difference + k * gap
    event_q = difference + k * next_gap
    pressure = (-ref.epi_weight * event_q, ref.epi_weight * event_q)
    rate = tuple(nu * p for nu, p in zip(capacity, pressure, strict=True))
    after = tuple(value + ref.timestep * velocity for value, velocity in zip(x, rate))
    multiplier = 1 - s * (2 * c + next_gap)
    after_difference = after[0] - after[1]
    after_q = after_difference + k * next_gap
    difference_identity = after_difference - (
        multiplier * difference - s * (2 * c + next_gap) * k * next_gap
    )
    q_identity = after_q - multiplier * (q - k * ref.coupling_factor * gap)
    mean_before, mean_after = sum(x) / 2, sum(after) / 2
    mean_change = mean_after - mean_before
    mean_flow = -s * next_gap * event_q / 2
    metric_before = _metric_mean(c, gap, x)
    metric_event = _metric_mean(c, next_gap, x)
    metric_after = _metric_mean(c, next_gap, after)
    metric_change = metric_after - metric_before
    metric_reset = (
        c
        * ref.coupling_factor
        * gap
        * difference
        / ((2 * c + gap) * (2 * c + next_gap))
    )
    before_domain = _inside(ref, gap, x)
    event_domain, after_domain = _inside(ref, next_gap, x), _inside(
        ref, next_gap, after
    )
    identities = (
        difference_identity,
        q_identity,
        mean_change - mean_flow,
        metric_change - metric_reset,
        metric_after - metric_event,
    )
    if any(identities):
        raise RuntimeError("exact P2 feedback lost its recurrence or mean identity")
    if before_domain and not (event_domain and after_domain):
        raise RuntimeError("exact P2 feedback left its proved invariant box")
    return P2CapacityFeedbackCycle(
        ref,
        gap,
        x,
        next_gap,
        capacity,
        after,
        (x[0] + k * (c + gap), x[1] + k * c),
        (x[0] + k * (c + next_gap), x[1] + k * c),
        (after[0] + k * (c + next_gap), after[1] + k * c),
        pressure,
        rate,
        (s * (c + next_gap), s * c),
        multiplier,
        difference,
        after_difference,
        q,
        event_q,
        after_q,
        difference_identity,
        q_identity,
        mean_before,
        mean_after,
        mean_change,
        mean_flow,
        mean_change - mean_flow,
        metric_before,
        metric_event,
        metric_after,
        metric_change,
        metric_reset,
        metric_change - metric_reset,
        metric_after - metric_event,
        before_domain,
        event_domain,
        after_domain,
        all(ref.epi_lower <= value <= ref.epi_upper for value in after),
    )


@dataclass(frozen=True)
class P2CapacityFeedbackBound:
    """Finite exact-model envelopes and a conditional limiting-mean interval."""

    reference: P2CapacityFeedbackReference
    capacity_gap: Fraction
    epi: Vector
    cycles: int
    capacity_gap_after: Fraction
    geometric_convolution: Fraction
    lifted_disagreement_bound: Fraction
    disagreement_bound: Fraction
    arithmetic_mean_tail_bound: Fraction
    limiting_mean_interval: Vector
    consensus_endpoint_bound: Fraction
    strict_disagreement_contraction: bool


def bound_p2_capacity_feedback(
    reference,
    *,
    capacity_gap,
    epi,
    cycles,
) -> P2CapacityFeedbackBound:
    """Bound an arbitrary finite repetition in the proved exact-model box.

    With a_n=rho**n*a0,
    |q_n| <= beta**n*|q0| + k*gamma*a0*beta*sum(beta**(n-1-j)*rho**j),
    and |d_n| <= |q_n|+k*a_n. The finite convolution handles beta=0 and beta=rho.
    The arithmetic mean always has an exact-model limit in this box, because
    |M_(n+1)-M_n| <= s*a_(n+1)*(U-ell)/2 is summable. The reported tail bounds
    its difference from M_n. beta<1 additionally makes both EPI coordinates
    converge to that mean; beta=1 does not provide that conclusion. No binary64
    asymptotic assertion, pressure refresh or repeated runtime admission follows.
    """
    ref, gap, x = _inputs(reference, capacity_gap, epi)
    if type(cycles) is not int or cycles < 0:
        raise ValueError("cycles must be a nonnegative integer")
    if not _inside(ref, gap, x):
        raise ValueError("a repetition envelope requires the initial invariant EPI box")
    beta, rho, k = ref.contraction_factor, ref.capacity_retention, ref.forcing_ratio
    beta_n, rho_n = beta**cycles, rho**cycles
    if cycles == 0:
        convolution = Fraction(0)
    elif beta == rho:
        convolution = cycles * beta ** (cycles - 1)
    else:
        convolution = (beta_n - rho_n) / (beta - rho)
    q0 = x[0] - x[1] + k * gap
    q_bound = beta_n * abs(q0) + k * ref.coupling_factor * gap * beta * convolution
    next_gap = gap * rho_n
    d_bound = q_bound + k * next_gap
    initial_tail = (
        ref.step_coefficient
        * (ref.epi_upper - ref.epi_lower)
        * gap
        * rho
        / (2 * ref.coupling_factor)
    )
    mean_tail = initial_tail * rho_n
    initial_mean = sum(x) / 2
    mean_interval = (
        max(ref.epi_lower, initial_mean - initial_tail),
        min(ref.epi_upper, initial_mean + initial_tail),
    )
    return P2CapacityFeedbackBound(
        ref,
        gap,
        x,
        cycles,
        next_gap,
        convolution,
        q_bound,
        d_bound,
        mean_tail,
        mean_interval,
        d_bound / 2 + mean_tail,
        ref.strict_disagreement_contraction,
    )


@dataclass(frozen=True)
class P2Binary64CouplingReference:
    """A proved rounded-capacity class; no graph or EPI recovery certificate."""

    coupling_factor: float
    exact_coupling_factor: Fraction
    lattice_step: Fraction
    max_index: int
    fixed_max_index: int
    relative_product_error: Fraction
    fixed_boundary_margin: Fraction
    descent_boundary_margin: Fraction
    product_upper_margin: Fraction
    proof_lower_gain: Fraction
    proof_lower_gain_margin: Fraction
    uniform_horizon: int
    uniform_horizon_verified: bool


def _binary64_platform():
    from .._binary64 import uses_ieee_binary64_rounding

    if not uses_ieee_binary64_rounding():
        raise ValueError(
            "coupling lattice requires IEEE binary64 round-to-nearest-even"
        )


def _float_input(value, name):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite built-in binary64 float")
    return Fraction.from_float(value)


def derive_p2_binary64_coupling_lattice(
    *, coupling_factor
) -> P2Binary64CouplingReference:
    """Prove the default-containing six-fixed-gap class without enumeration.

    The map is RN(v+RN(gamma*RN(1-v))) with separate binary64 operations, no FMA,
    and 1<=v<=2. Write v=1+j*delta, delta=2**-52, 0<=j<=2**52. Sterbenz makes
    the subtraction exact. Every positive product is normal, so its relative
    rounding error is at most u=2**-53. The validated strict inequalities
    6*gamma*(1+u)<1/2 and 7*gamma*(1-u)>1/2 prove precisely j=0,...,6 fixed.

    For j>=7 the index strictly descends. Since gamma<1/8 and j*delta/8 is
    representable, monotonic rounding bounds the rounded product by j*delta/8.
    Thus j>=8 cannot jump below 7, and j=7 maps to 6. Final halfway cases round
    the resulting capacity to even; they are not evaluated by separately
    rounding a decrement. The boundary v=2 has its output inside the same grid.

    The lower product gain exceeds 5/64, giving j_next<=(59/64)*j+1/2. The
    integer inequality 59**512*(5*2**52-32)<3*64**512 proves the upper envelope
    is below 7 after 512 steps. This is an analytic bound, not an executed or
    chosen runtime horizon. Every initial index above 6 therefore reaches 6;
    smaller indices stay fixed. Other factors are accepted only if these same
    explicit class inequalities hold. No EPI-error lower bound follows.
    """
    _binary64_platform()
    gain = _float_input(coupling_factor, "coupling_factor")
    unit = Fraction(1, 2**53)
    fixed_margin = Fraction(1, 2) - 6 * gain * (1 + unit)
    descent_margin = 7 * gain * (1 - unit) - Fraction(1, 2)
    upper_margin = Fraction(1, 8) - gain
    lower_gain = Fraction(5, 64)
    lower_margin = gain * (1 - unit) - lower_gain
    if min(fixed_margin, descent_margin, upper_margin, lower_margin) <= 0:
        raise ValueError(
            "coupling factor does not belong to the proved six-fixed-gap class"
        )
    spacing, maximum, horizon = Fraction(1, 2**52), 2**52, 512
    if gain * spacing < Fraction(1, 2**1022):
        raise RuntimeError("the coupling lattice normal-product premise was lost")
    horizon_verified = 59**horizon * (5 * maximum - 32) < 3 * 64**horizon
    if not horizon_verified:
        raise RuntimeError("the uniform coupling-lattice horizon inequality failed")
    return P2Binary64CouplingReference(
        coupling_factor,
        gain,
        spacing,
        maximum,
        6,
        unit,
        fixed_margin,
        descent_margin,
        upper_margin,
        lower_gain,
        lower_margin,
        horizon,
        horizon_verified,
    )


def _binary64_reference(value):
    if type(value) is not P2Binary64CouplingReference:
        raise TypeError("reference must be a P2Binary64CouplingReference")
    return derive_p2_binary64_coupling_lattice(coupling_factor=value.coupling_factor)


@dataclass(frozen=True)
class P2Binary64CouplingObservation:
    """One reproduced numeric transition and a conditional kernel-only limit."""

    reference: P2Binary64CouplingReference
    capacity_before: float
    capacity_after: float
    index_before: int
    index_after: int
    index_decrease: int
    is_fixed: bool
    eventual_index: int
    eventual_capacity: float
    eventual_gap: Fraction
    termination_horizon_from_before: int
    ideal_capacity_gap_after: Fraction
    capacity_gap_defect: Fraction


def observe_p2_binary64_coupling_lattice(
    reference,
    *,
    capacity_before,
    capacity_after,
) -> P2Binary64CouplingObservation:
    """Match a supplied endpoint to the shared production capacity kernel.

    Both inputs must be built-in finite floats in [1,2]; the neighbor is exactly
    1.0. The rounded result must equal the production helper, including its
    fsum/singleton mean, subtraction, multiplication and addition order. Equality
    verifies these detached numeric inputs only, not the caller's graph, grammar,
    history, pressure refresh, platform provenance or future execution. The
    eventual index concerns repetition of this restricted numeric map alone.
    The signed gap defect compares one represented step with (1-gamma)*a.
    """
    ref = _binary64_reference(reference)
    before = _float_input(capacity_before, "capacity_before")
    after = _float_input(capacity_after, "capacity_after")
    if not 1 <= before <= 2 or not 1 <= after <= 2:
        raise ValueError("coupling lattice capacities must lie in [1,2]")
    initial_index, following_index = (before - 1) / ref.lattice_step, (
        after - 1
    ) / ref.lattice_step
    if initial_index.denominator != 1 or following_index.denominator != 1:
        raise ValueError("capacities must belong to the declared unit-binade lattice")
    from ..operators._coupling_stage_kernel import coupling_capacity_blend

    expected = coupling_capacity_blend(capacity_before, (1.0,), ref.coupling_factor)
    if capacity_after != expected:
        raise ValueError(
            "observed capacity endpoint differs from the shared production kernel"
        )
    j, following = int(initial_index), int(following_index)
    if (
        (j <= ref.fixed_max_index and following != j)
        or (j > ref.fixed_max_index and not ref.fixed_max_index <= following < j)
        or (j >= 8 and following < 7)
    ):
        raise RuntimeError(
            "production capacity transition violated the proved lattice class"
        )
    terminal = min(j, ref.fixed_max_index)
    terminal_gap = terminal * ref.lattice_step
    ideal_gap = (1 - ref.exact_coupling_factor) * (before - 1)
    return P2Binary64CouplingObservation(
        ref,
        capacity_before,
        capacity_after,
        j,
        following,
        j - following,
        j == following,
        terminal,
        float(1 + terminal_gap),
        terminal_gap,
        0 if j <= ref.fixed_max_index else ref.uniform_horizon,
        ideal_gap,
        after - 1 - ideal_gap,
    )
