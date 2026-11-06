"""Exact detached budgets for two declared passive contact completions.

The fully weighted nodal model in ``RELATION_FOUNDATIONS.md`` supplies
``S_dot = -sum(D_i) + sum(c_e * a_dot_e)``. This module allocates its admitted
nonnegative losses to candidate contacts without spending a node's loss twice.
It also bounds one declared near-consensus source domain with an exact
rational majorant of its common Lyapunov metric, and computes all-time retained
response bounds for one declared seed family. These readers authenticate no
graph and install no weighted phase law or support executor.
They execute no structural operator and authenticate no grammar or U3 gate.
Positive contact rate is not evidence that Coupling (UM) admits the candidate.
The caller must establish that losses, costs and storage belong to the same
complete model. Exact accounting of supplied scalars is not that model proof.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from operator import index

from .._exact_time import exact_or_represented_real
from ._cycle_algebra import Vector, ordered_vector

__all__ = (
    "PassiveContactBudget",
    "PassiveContactGrowthBound",
    "PassiveContactConsensusBound",
    "PassiveContactSeedResponseBound",
    "evaluate_passive_contact_budget",
    "bound_passive_contact_growth",
    "bound_passive_contact_consensus_source",
    "bound_passive_contact_seed_response",
)


def _ordered(values, label):
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be ordered")
    try:
        return tuple(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be ordered") from exc


def _nonnegative(values, label):
    result = ordered_vector(values, label)
    if any(value < 0 for value in result):
        raise ValueError(f"{label} must be nonnegative")
    return result


def _positive_scale(value):
    beta = exact_or_represented_real(value, "storage_scale")
    if beta <= 0:
        raise ValueError("storage_scale must be positive")
    return beta


@dataclass(frozen=True)
class PassiveContactBudget:
    """Exact instantaneous ledger, not a trajectory or graph certificate."""

    node_losses: Vector
    candidates: tuple[tuple[int, int], ...]
    contact_costs: Vector
    port_shares: tuple[Vector, ...]
    storage_scale: Fraction
    law: str
    allocated_node_fractions: Vector
    contact_allocations: Vector
    contact_rates: Vector
    total_loss: Fraction
    unallocated_loss: Fraction
    support_work: Fraction
    storage_rate: Fraction
    accounting_rate: Fraction


def evaluate_passive_contact_budget(
    *,
    node_losses,
    candidates,
    contact_costs,
    port_shares,
    storage_scale,
    law="reciprocal",
) -> PassiveContactBudget:
    """Allocate loss locally and compute one of the two existing support rows.

    ``candidates[e]=(u,v)`` indexes ``node_losses``. Its two ``port_shares``
    allocate fractions of those endpoint losses, even when a current loss is
    zero. Each node's total allocated fraction must be at most one. Candidates
    are distinct unordered pairs of distinct nodes; their supplied order and
    endpoint orientation are retained. Empty candidate lists are permitted.

    The declared rows are ``B/(beta+c)`` (``reciprocal``) and
    ``beta*B/(beta+c)**2`` (``squared_reciprocal``). ``accounting_rate`` is the
    derivative of the auxiliary ``S + beta*sum(a_e)`` under the same balance;
    it does not add a storage reservoir or change the nodal law. This is an
    instantaneous calculation. Future allocation/domain validity remains a
    separate premise of any integrated claim.
    """
    losses = _nonnegative(node_losses, "node_losses")
    costs = _nonnegative(contact_costs, "contact_costs")
    beta = _positive_scale(storage_scale)
    if not isinstance(law, str) or law not in (
        "reciprocal",
        "squared_reciprocal",
    ):
        raise ValueError("law must name one of the two passive completions")
    raw_pairs = _ordered(candidates, "candidates")
    shares = tuple(
        _nonnegative(row, "port_shares row")
        for row in _ordered(port_shares, "port_shares")
    )
    if len(raw_pairs) != len(costs) or len(shares) != len(costs):
        raise ValueError("candidate, cost and share counts must agree")
    pairs, seen = [], set()
    spent = [Fraction(0) for _ in losses]
    allocations = []
    for raw, share in zip(raw_pairs, shares, strict=True):
        endpoints = _ordered(raw, "candidate")
        if len(endpoints) != 2 or len(share) != 2:
            raise ValueError("each candidate and share row must have two entries")
        pair = []
        for value in endpoints:
            if isinstance(value, bool):
                raise TypeError("candidate indices must be integers, not booleans")
            try:
                item = index(value)
            except TypeError as exc:
                raise TypeError("candidate indices must be integers") from exc
            if not 0 <= item < len(losses):
                raise ValueError("candidate index outside node_losses")
            pair.append(item)
        u, v = pair
        key = tuple(sorted(pair))
        if u == v or key in seen:
            raise ValueError("candidates must be distinct pairs of distinct nodes")
        seen.add(key)
        pairs.append((u, v))
        spent[u] += share[0]
        spent[v] += share[1]
        allocations.append(share[0] * losses[u] + share[1] * losses[v])
    if any(value > 1 for value in spent):
        raise ValueError("combined contact shares overspend a node's loss")
    rates = tuple(
        (
            amount / (beta + cost)
            if law == "reciprocal"
            else beta * amount / (beta + cost) ** 2
        )
        for amount, cost in zip(allocations, costs, strict=True)
    )
    total = sum(losses, Fraction(0))
    unused = total - sum(allocations, Fraction(0))
    work = sum((c * rate for c, rate in zip(costs, rates, strict=True)), Fraction(0))
    storage_rate = -total + work
    accounting_rate = storage_rate + beta * sum(rates, Fraction(0))
    return PassiveContactBudget(
        losses,
        tuple(pairs),
        costs,
        shares,
        beta,
        law,
        tuple(spent),
        tuple(allocations),
        rates,
        total,
        unused,
        work,
        storage_rate,
        accounting_rate,
    )


@dataclass(frozen=True)
class PassiveContactGrowthBound:
    """Conditional resource exclusion; a nonexcluded request is undecided."""

    initial_storage: Fraction
    storage_scale: Fraction
    initial_weights: Vector
    sum_increase_upper_bound: Fraction
    individual_weight_upper_bounds: Vector
    required_increases: Vector | None
    resource_excluded: bool | None
    background_strengths: Vector | None
    resultant_real_lower_bounds: Vector | None


def bound_passive_contact_growth(
    *,
    initial_storage,
    storage_scale,
    initial_weights,
    required_increases=None,
    background_strengths=None,
) -> PassiveContactGrowthBound:
    """Bound contact growth for a continued admitted reciprocal completion.

    Requires nonnegative storage and weights, the complete shared balance,
    held beta, and admissible nonoverlapping loss allocations throughout the
    interval. Either implemented support row gives
    ``sum(a_e(t)-a_e(0)) <= S(0)/beta``. This function admits the initial
    scalars and computes that bound; it does not certify those future premises.

    Optional requirements are necessary nonnegative increases specified by a
    separate identity criterion. Their sum strictly exceeding the budget
    excludes that requirement. Equality or a smaller sum proves no acquisition.
    No requirements means ``resource_excluded=None``, not a passing verdict.

    Optional strictly positive fixed background strengths give the conditional
    bounds ``Re Z_i >= d_background_i - S(0)/beta`` for the same fully weighted
    storage. Strict positivity of every bound suffices to retain the regular
    right-half-plane chart. A zero/negative bound is inconclusive, not proof of
    singularity. No graph identity, actual initial storage, phase row or global
    continuation premise is authenticated by this scalar calculation.
    """
    storage = exact_or_represented_real(initial_storage, "initial_storage")
    if storage < 0:
        raise ValueError("initial_storage must be nonnegative")
    beta = _positive_scale(storage_scale)
    weights = _nonnegative(initial_weights, "initial_weights")
    requirements = None
    excluded = None
    strengths = None
    resultant_bounds = None
    budget = storage / beta
    if background_strengths is not None:
        strengths = _nonnegative(background_strengths, "background_strengths")
        if not strengths or any(value <= 0 for value in strengths):
            raise ValueError("background_strengths must be nonempty and positive")
        resultant_bounds = tuple(value - budget for value in strengths)
    if required_increases is not None:
        requirements = _nonnegative(required_increases, "required_increases")
        if len(requirements) != len(weights):
            raise ValueError("requirement and initial weight counts must agree")
        excluded = sum(requirements, Fraction(0)) > budget
    return PassiveContactGrowthBound(
        storage,
        beta,
        weights,
        budget,
        tuple(a + budget for a in weights),
        requirements,
        excluded,
        strengths,
        resultant_bounds,
    )


@dataclass(frozen=True)
class PassiveContactConsensusBound:
    """A conditional small-source bound, not an acquisition certificate."""

    form_deviations: Vector
    phase_lift_deviations: Vector
    background_strengths: Vector
    metric_upper_bound: Fraction
    strict_threshold: Fraction
    within_sufficient_ball: bool


def bound_passive_contact_consensus_source(
    *,
    form_deviations,
    phase_lift_deviations,
    background_strengths,
) -> PassiveContactConsensusBound:
    """Bound the common-metric source ball for one declared parameter class.

    The conditional theorem fixes ``e=w=1/2`` and ``beta=nu_i=1`` in its
    declared structural units, with all candidate weights initially zero.
    Deviations use one FIXED common form/phase reference and real phase lifts;
    this reader never wraps phases or constructs a moving mean. Strengths
    belong to the same fixed symmetric nonnegative background.

    Its exact Lyapunov matrix is ``P=((2,pi),(pi,2+pi**2))``. Using
    ``pi<22/7`` and ``pi**2<10`` bounds initial ``sum(d_i*z_i.T*P*z_i)``
    by ``sum(d_i*(2*x_i**2+(44/7)*abs(x_i*t_i)+12*t_i**2))``.
    Strictly less than ``min(d_i)/256`` is sufficient for the radius-1/16
    trapping theorem under either existing passive completion and admissible
    fixed loss shares. Equality or failure is inconclusive, not an exclusion.

    This function admits and computes scalar premises; it does not verify
    graph strengths, zero initial contacts, law coefficients or phase lifts.
    It does not establish positive onset, connectedness after onset, isolated
    formation or actual-source entry. Those remain additional obligations of
    an acquisition claim. No weighted nonlinear field or trajectory is run.
    """
    forms = ordered_vector(form_deviations, "form_deviations")
    phases = ordered_vector(phase_lift_deviations, "phase_lift_deviations")
    strengths = _nonnegative(background_strengths, "background_strengths")
    if not strengths or any(value <= 0 for value in strengths):
        raise ValueError("background_strengths must be nonempty and positive")
    if len(forms) != len(strengths) or len(phases) != len(strengths):
        raise ValueError("form, phase and background strength counts must agree")
    upper = sum(
        (
            strength
            * (2 * form**2 + Fraction(44, 7) * abs(form * phase) + 12 * phase**2)
            for form, phase, strength in zip(forms, phases, strengths, strict=True)
        ),
        Fraction(0),
    )
    threshold = min(strengths) / 256
    return PassiveContactConsensusBound(
        forms, phases, strengths, upper, threshold, upper < threshold
    )


@dataclass(frozen=True)
class PassiveContactSeedResponseBound:
    """Conditional retained-response bounds for the declared four-node seed."""

    amplitude: Fraction
    leading_contact: Fraction
    contact_error_bound: Fraction
    contact_bounds: tuple[Fraction, Fraction]
    susceptibility_bounds: tuple[Fraction, Fraction]
    receiver_gain_bounds: tuple[Fraction, Fraction]
    tangent_decay_rate_bounds: tuple[Fraction, Fraction]


def bound_passive_contact_seed_response(
    *, amplitude
) -> PassiveContactSeedResponseBound:
    """Bound an all-time limit analytically, without evaluating a trajectory.

    Requires the declared fully weighted passive law on two unit P2
    backgrounds (0,1),(2,3), candidate (1,2), e=w=1/2, beta=nu_i=1,
    x=(0,amplitude,0,0), zero phase/contact and full endpoint loss allocation.
    Either existing reciprocal support row obeys the same enclosure; this
    does not assert identical destinations or higher-order response.

    For |amplitude|<=1/32, all-time loss accounting proves
    |a_infinity-amplitude**2/4| <= 2*|amplitude|**3 + 3*amplitude**4
    + 3*amplitude**6. The interval also retains the independent storage bound
    0<=a_infinity<=amplitude**2/2. Nonzero admitted amplitudes have a strictly
    positive lower bound. Zero is an exactly stationary no-contact control.
    At zero, the returned zero rate denotes additional neutral inter-block
    modes; it is not a connected-family acquisition or recovery certificate.

    Susceptibility is chi=a/(1+a); receiver form-rate gain is chi/2.
    The consensus tangent's slow transverse decay rate is kappa*chi, with
    kappa=(1-sqrt(1-4/pi**2))/4 in (1/20,1/15). Its returned rational bounds
    describe that local linear exponent, not a global arrival deadline or
    nonlinear recovery prefactor. Common shifts/relation modes remain neutral.

    Only amplitude is admitted here. The function does not authenticate a
    graph, preparation history, units or law; different phases, weights,
    parameters or source shapes require a separate theorem. An out-of-domain
    amplitude is rejected, not classified as incapable of acquisition.
    """
    epsilon = exact_or_represented_real(amplitude, "amplitude")
    magnitude = abs(epsilon)
    if magnitude > Fraction(1, 32):
        raise ValueError("amplitude must satisfy abs(amplitude) <= 1/32")
    leading = epsilon**2 / 4
    error = 2 * magnitude**3 + 3 * epsilon**4 + 3 * epsilon**6
    contact = (max(Fraction(0), leading - error), min(epsilon**2 / 2, leading + error))
    response = tuple(value / (1 + value) for value in contact)
    gain = tuple(value / 2 for value in response)
    decay = (response[0] / 20, response[1] / 15)
    return PassiveContactSeedResponseBound(
        epsilon, leading, error, contact, response, gain, decay
    )
