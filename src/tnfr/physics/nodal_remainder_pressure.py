"""Exact carried nodal pressure and accumulated-source diagnostics.

The graph displays x while the numerical representation retains X=x+r.
This detached diagnostic separates that readout difference from everything
else in supplied stored pressure. It does not identify a pressure producer,
an operator event, a graph trajectory or the accuracy of the phase channel.
Periodic and finite-class budgets retain their separate source hypotheses;
affine area crossings concern supplied held inputs and a declared prefix.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm

from ..dynamics._euler_kernel import (
    NodalRemainderState,
    _binary64_tuple,
    _finite_binary64,
    _require_remainder_rounding,
    _validate_nodal_remainder_state,
)
from ._cycle_algebra import Vector, dot, laplacian_action
from ._exact_linear_algebra import ExactSquareMatrix, _require_exact_square_matrix
from .binary64_nodal_flow import _rounding_cell
from .support_transport import _laplacian

__all__ = [
    "NodalRemainderPressureReadout",
    "observe_nodal_remainder_pressure_readout",
    "PeriodicPhaseSourceBudget",
    "PeriodicPhaseSourceCompensation",
    "derive_periodic_phase_source_budget",
    "observe_periodic_phase_source_compensation",
    "FiniteNodalPressureDrift",
    "observe_finite_nodal_pressure_drift",
    "NodalAreaCrossing",
    "NodalAreaCrossings",
    "derive_nodal_area_crossings",
    "TwoLevelNodalReturn",
    "derive_two_level_nodal_return",
    "FiniteLevelNodalReturn",
    "derive_finite_level_nodal_return",
    "NodalRemainderCycleGradient",
    "observe_nodal_remainder_cycle_gradient",
]


def _mean(values: Vector) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


@dataclass(frozen=True, slots=True)
class NodalRemainderCycleGradient:
    """Exact cycle-index change resolved into nodal area and carry transfer.

    The full supplied area must equal reconstructed endpoint change before
    the Laplacian is applied. Its constant null mode cannot hide a false
    uniform area. This is an endpoint identity, not pressure provenance,
    numerical trajectory evidence or a realizable transition certificate.
    """

    gradient_indices_before: tuple[int, ...]
    gradient_indices_after: tuple[int, ...]
    gradient_index_change: Vector
    nodal_area: Vector
    remainder_change: Vector
    nodal_gradient_term: Vector
    remainder_gradient_term: Vector
    identity_residual: Vector

    @property
    def pressure_provenance_certified(self) -> bool:
        return False

    @property
    def runtime_provenance_certified(self) -> bool:
        return False


def observe_nodal_remainder_cycle_gradient(
    *,
    initial: NodalRemainderState,
    endpoint: NodalRemainderState,
    nodal_area: Vector,
    epi_quantum: Fraction,
) -> NodalRemainderCycleGradient:
    """Check full nodal area, then derive an ordered cycle's gradient budget.

    Write X=x+r, A=X_after-X_before, and m=-2*L_rw*x/delta on a
    simple unit-conductance cycle. Then
    m_after-m_before=-2*L_rw*A/delta+2*L_rw*(r_after-r_before)/delta.
    Integral endpoint indices are required. The full-vector area identity
    is checked first: L_rw alone would miss a uniform discrepancy in A.
    Exact retained endpoints do not authenticate how their change arose.
    """
    _require_remainder_rounding()
    start = _validate_nodal_remainder_state(initial)
    end = _validate_nodal_remainder_state(endpoint)
    size = len(start)
    if size < 3 or len(end) != size:
        raise ValueError("cycle endpoints must have the same size of at least three")
    if type(epi_quantum) is not Fraction or epi_quantum <= 0:
        raise ValueError("epi_quantum must be a strictly positive Fraction")
    if (
        type(nodal_area) is not tuple
        or len(nodal_area) != size
        or any(type(value) is not Fraction for value in nodal_area)
    ):
        raise TypeError("nodal_area must be a matching tuple of exact Fractions")
    if nodal_area != tuple(b - a for a, b in zip(start, end, strict=True)):
        raise ValueError("nodal_area must equal the full reconstructed endpoint change")
    before = tuple(
        -2 * value / epi_quantum
        for value in laplacian_action(tuple(map(Fraction, initial.epi)))
    )
    after = tuple(
        -2 * value / epi_quantum
        for value in laplacian_action(tuple(map(Fraction, endpoint.epi)))
    )
    if any(value.denominator != 1 for value in before + after):
        raise ValueError(
            "the endpoint gradients must belong to the declared integer lattice"
        )
    change = tuple(b - a for a, b in zip(before, after, strict=True))
    carry = tuple(
        b - a for a, b in zip(initial.remainder, endpoint.remainder, strict=True)
    )
    nodal_term = tuple(
        -2 * value / epi_quantum for value in laplacian_action(nodal_area)
    )
    carry_term = tuple(2 * value / epi_quantum for value in laplacian_action(carry))
    residual = tuple(
        delta - area - transfer
        for delta, area, transfer in zip(change, nodal_term, carry_term, strict=True)
    )
    if any(residual):
        raise RuntimeError("the cycle gradient lost its exact nodal/carry identity")
    return NodalRemainderCycleGradient(
        tuple(map(int, before)),
        tuple(map(int, after)),
        change,
        nodal_area,
        carry,
        nodal_term,
        carry_term,
        residual,
    )


@dataclass(frozen=True, slots=True)
class FiniteNodalPressureDrift:
    """Exact separating functional for a supplied finite visible-state class.

    Each visible tuple represents every admissible carry in its product of
    nearest-even cells. Closed cell envelopes make the width conservative
    at odd ties. The pressure map's provenance is a separate obligation.
    The bound concerns exit from this finite class, not from the EPI band.
    """

    epi_states: tuple[tuple[float, ...], ...]
    pressure_vectors: tuple[tuple[float, ...], ...]
    functional: Vector
    timestep: float
    epi_lower: float
    epi_upper: float
    functional_projections: Vector
    minimum_pressure_projection: Fraction
    state_functional_lower: Vector
    state_functional_upper: Vector
    class_lower: Fraction
    class_upper: Fraction
    width: Fraction
    max_confined_steps: int | None
    escape_step_bound: int | None

    @property
    def conditional_class_escape_certified(self) -> bool:
        return self.minimum_pressure_projection > 0

    @property
    def pressure_provenance_certified(self) -> bool:
        return False

    @property
    def positive_band_exit_certified(self) -> bool:
        return False


def observe_finite_nodal_pressure_drift(
    *,
    epi_states: tuple[tuple[float, ...], ...],
    pressure_vectors: tuple[tuple[float, ...], ...],
    functional: Vector,
    timestep: float,
    epi_lower: float = 0.05,
    epi_upper: float = 1.0,
) -> FiniteNodalPressureDrift:
    """Bound residence in a finite class by a strictly positive projection.

    Assume unit capacities, fixed h>0, no EPI jumps, retained carry and a
    deterministic pressure map assigning the supplied p(x) to every listed
    visible tuple. If c=min_x dot(ell,p(x))>0, exact nodal evolution gives
    dot(ell,X_N-X_0)>=N*h*c while all pre/post states remain in the class.
    The rounding-cell enclosure supplies its total functional width W;
    hence N<=floor(W/(h*c)). The next step must leave this class or fail
    the declared update contract. Nonpositive c is inconclusive.

    All admissible carry values at every supplied visible state are covered.
    This does not assume repeated visible values mean repeated exact X.
    The caller must separately identify the actual pressure producer and
    fixed phase/configuration. No graph, transition, reachability, positive
    band exit or future runtime admission is certified by these inputs.
    A separating functional is an observation; it never changes pressure.
    """
    _require_remainder_rounding()
    if type(epi_states) is not tuple or type(pressure_vectors) is not tuple:
        raise TypeError("the finite state/pressure classes must be ordered tuples")
    if not epi_states or len(pressure_vectors) != len(epi_states):
        raise ValueError("a nonempty state class needs one pressure vector per state")
    states = tuple(_binary64_tuple(row, "epi_states") for row in epi_states)
    pressures = tuple(
        _binary64_tuple(row, "pressure_vectors") for row in pressure_vectors
    )
    size = len(states[0])
    if any(len(row) != size for row in states + pressures):
        raise ValueError(
            "every visible state and pressure must have the same dimension"
        )
    if len({tuple(value.hex() for value in row) for row in states}) != len(states):
        raise ValueError("visible states must be distinct in the supplied pressure map")
    if (
        type(functional) is not tuple
        or len(functional) != size
        or any(type(value) is not Fraction for value in functional)
    ):
        raise TypeError("functional must be a matching tuple of exact Fraction values")
    if not any(functional):
        raise ValueError("the separating functional must be nonzero")
    h = _finite_binary64(timestep, "timestep")
    lower = _finite_binary64(epi_lower, "epi_lower")
    upper = _finite_binary64(epi_upper, "epi_upper")
    if h <= 0 or not 0 < lower <= upper <= 1:
        raise ValueError("positive timestep and a positive unit EPI band are required")
    if any(not lower <= value <= upper for row in states for value in row):
        raise ValueError("every displayed EPI must belong to the declared band")
    exact_lower, exact_upper = Fraction(lower), Fraction(upper)
    lower_bounds, upper_bounds = [], []
    for row in states:
        cells = tuple(_rounding_cell(value, Fraction(value)) for value in row)
        intervals = tuple(
            (max(cell.lower, exact_lower), min(cell.upper, exact_upper))
            for cell in cells
        )
        lower_bounds.append(
            sum(
                (
                    weight * (left if weight >= 0 else right)
                    for weight, (left, right) in zip(functional, intervals, strict=True)
                ),
                Fraction(0),
            )
        )
        upper_bounds.append(
            sum(
                (
                    weight * (right if weight >= 0 else left)
                    for weight, (left, right) in zip(functional, intervals, strict=True)
                ),
                Fraction(0),
            )
        )
    projections = tuple(dot(functional, tuple(map(Fraction, row))) for row in pressures)
    gap = min(projections)
    class_lower, class_upper = min(lower_bounds), max(upper_bounds)
    width = class_upper - class_lower
    maximum = width // (Fraction(h) * gap) if gap > 0 else None
    return FiniteNodalPressureDrift(
        states,
        pressures,
        functional,
        h,
        lower,
        upper,
        projections,
        gap,
        tuple(lower_bounds),
        tuple(upper_bounds),
        class_lower,
        class_upper,
        width,
        maximum,
        maximum + 1 if maximum is not None else None,
    )


@dataclass(frozen=True, slots=True)
class NodalRemainderPressureReadout:
    """Supplied-state EPI diffusion references and exact signed differences.

    ``stored_minus_visible_reference`` includes all non-EPI channels, their
    realization, stale pressure and operator writes. It is not identified
    with numerical error. The reversible weighted mean concerns the nodal
    readout shift nu*w*L_rw*r, not the pressure shift alone. Its weights are
    undefined here if any capacity is zero; those fields are then None.
    Public fields are detached data, not authenticated runtime evidence.
    """

    state: NodalRemainderState
    conductance: ExactSquareMatrix
    capacity: Vector
    stored_pressure: Vector
    epi_weight: Fraction
    strengths: Vector
    epi_pressure_visible: Vector
    epi_pressure_reconstructed: Vector
    readout_shift: Vector
    stored_minus_visible_reference: Vector
    stored_minus_reconstructed_reference: Vector
    pressure_identity_residual: Vector
    nodal_readout_shift: Vector
    mean_pressure_readout_shift: Fraction
    mean_nodal_readout_shift: Fraction
    degree_weighted_pressure_readout_shift: Fraction
    reversible_weights: Vector | None
    reversible_mean_nodal_readout_shift: Fraction | None


def observe_nodal_remainder_pressure_readout(
    *,
    state: NodalRemainderState,
    conductance: ExactSquareMatrix,
    capacity: tuple[float, ...],
    stored_pressure: tuple[float, ...],
    epi_weight: float | Fraction,
) -> NodalRemainderPressureReadout:
    """Compare pure EPI diffusion at visible x and reconstructed X=x+r.

    The supplied symmetric nonnegative exact conductance must have positive
    row strengths d_i. Zero-strength rows are outside this bounded observer;
    disconnected positive-degree components and self-loops are permitted.
    Capacities and stored pressures must be actual finite binary64 floats.
    Capacities and the exact or represented EPI coefficient are nonnegative.
    Every public encoding field is validated before its remainder is used.

    For L_rw=D^-1(D-W), p_epi(x)=-w*L_rw*x and p_epi(X)=-w*L_rw*X,
    the exact readout shift p_epi(x)-p_epi(X)=w*L_rw*r gives
    stored-p_epi(X)=(stored-p_epi(x))+w*L_rw*r. No pure-EPI assumption
    is made about stored pressure, and no nonlinear phase ideal is inferred.

    Symmetry gives sum(d_i*shift_i)=0. With all nu_i>0, H_i=d_i/nu_i
    therefore gives sum(H_i*nu_i*shift_i)=0. Regular conductance and a
    common capacity also give zero arithmetic mean nodal shift. Unequal
    capacities or irregular degree generally prevent that arithmetic
    cancellation. These are instantaneous identities; changing the metric
    between events does not establish a conserved weighted trajectory mean.
    """
    if not isinstance(state, NodalRemainderState):
        raise TypeError("state must be a NodalRemainderState")
    exact = _validate_nodal_remainder_state(state)
    visible = tuple(Fraction.from_float(value) for value in state.epi)
    matrix = _require_exact_square_matrix(conductance, name="conductance")
    size = len(visible)
    if len(matrix) != size:
        raise ValueError("conductance must match the EPI dimensions")
    if any(value < 0 for row in matrix for value in row):
        raise ValueError("conductance must be nonnegative")
    if any(matrix[i][j] != matrix[j][i] for i in range(size) for j in range(size)):
        raise ValueError("conductance must be symmetric")
    strengths = tuple(sum(row, Fraction(0)) for row in matrix)
    if any(value <= 0 for value in strengths):
        raise ValueError("conductance rows must have positive strength")
    capacities = _binary64_tuple(capacity, "capacity")
    pressures = _binary64_tuple(stored_pressure, "stored_pressure")
    if len(capacities) != size or len(pressures) != size:
        raise ValueError("capacity and stored pressure must match the EPI dimensions")
    nu = tuple(Fraction.from_float(value) for value in capacities)
    pressure = tuple(Fraction.from_float(value) for value in pressures)
    if any(value < 0 for value in nu):
        raise ValueError("capacity must be nonnegative")
    weight = (
        epi_weight
        if type(epi_weight) is Fraction
        else Fraction.from_float(_finite_binary64(epi_weight, "epi_weight"))
    )
    if weight < 0:
        raise ValueError("epi_weight must be nonnegative")
    edges = tuple(
        (i, j, value)
        for i, row in enumerate(matrix)
        for j, value in enumerate(row)
        if value
    )

    def laplacian(values):
        return tuple(
            value / degree
            for value, degree in zip(_laplacian(edges, values), strengths, strict=True)
        )

    epi_visible = tuple(-weight * value for value in laplacian(visible))
    epi_exact = tuple(-weight * value for value in laplacian(exact))
    shift = tuple(weight * value for value in laplacian(state.remainder))
    residual_visible = tuple(
        p - value for p, value in zip(pressure, epi_visible, strict=True)
    )
    residual_exact = tuple(
        p - value for p, value in zip(pressure, epi_exact, strict=True)
    )
    identity = tuple(
        total - other - changed
        for total, other, changed in zip(
            residual_exact, residual_visible, shift, strict=True
        )
    )
    nodal_shift = tuple(v * value for v, value in zip(nu, shift, strict=True))
    degree_mean = dot(strengths, shift) / sum(strengths, Fraction(0))
    reversible = (
        tuple(d / v for d, v in zip(strengths, nu, strict=True)) if all(nu) else None
    )
    reversible_mean = (
        dot(reversible, nodal_shift) / sum(reversible, Fraction(0))
        if reversible is not None
        else None
    )
    if any(identity) or degree_mean != 0 or reversible_mean not in (None, Fraction(0)):
        raise RuntimeError(
            "pressure readout lost its exact decomposition or reversible balance"
        )
    return NodalRemainderPressureReadout(
        state,
        matrix,
        nu,
        pressure,
        weight,
        strengths,
        epi_visible,
        epi_exact,
        shift,
        residual_visible,
        residual_exact,
        identity,
        nodal_shift,
        _mean(shift),
        _mean(nodal_shift),
        degree_mean,
        reversible,
        reversible_mean,
    )


@dataclass(frozen=True, slots=True)
class PeriodicPhaseSourceBudget:
    """Exact periodic mean decomposition of supplied weighted C6 sources.

    Each six-coordinate row is the represented weighted phase contribution
    A before EPI/channel assembly, held over one block of duration H with
    unit capacity. The caller must independently establish that these rows
    form the source cycle of the phase map. Neither period closure nor
    production provenance follows from this detached algebraic reference.
    Centered prefixes are in pressure units; their amplitude bound includes H.
    """

    phase_contributions: tuple[tuple[float, ...], ...]
    block_duration: float
    period: int
    mean_sources: Vector
    mean_source: Fraction
    centered_prefixes: Vector
    phase_area_per_period: Fraction
    prefix_amplitude_bound: Fraction

    @property
    def phase_cycle_certified(self) -> bool:
        return False


def derive_periodic_phase_source_budget(
    *,
    phase_contributions: tuple[tuple[float, ...], ...],
    block_duration: float,
) -> PeriodicPhaseSourceBudget:
    """Split a declared periodic source into constant drift and finite offsets.

    For period m, row means a_j, b=sum(a_j)/m and C_r=sum_(j<r)(a_j-b),
    C_0=C_m=0. Across N complete blocks of equal represented duration H,
    phase_area(N)=H*(N*b+C_(N mod m)). Its periodic offset has magnitude
    at most H*max_r(abs(C_r)). No iteration over N or floating summation is
    needed. This reference neither generates phases nor proves the supplied
    cycle occurs. A zero average b controls this phase-source contribution
    only; the full nodal mean still includes EPI-reduction and assembly terms.
    """
    if type(phase_contributions) is not tuple:
        raise TypeError(
            "phase_contributions must be an ordered tuple of six-coordinate rows"
        )
    if not phase_contributions:
        raise ValueError("the declared phase-source cycle must be nonempty")
    sources = tuple(
        _binary64_tuple(row, f"phase_contributions[{i}]")
        for i, row in enumerate(phase_contributions)
    )
    if any(len(row) != 6 for row in sources):
        raise ValueError(
            "every phase-source row must contain the six ordered C6 coordinates"
        )
    duration = _finite_binary64(block_duration, "block_duration")
    if duration <= 0:
        raise ValueError("block_duration must be positive")
    exact_duration = Fraction.from_float(duration)
    means = tuple(
        _mean(tuple(Fraction.from_float(value) for value in row)) for row in sources
    )
    drift = _mean(means)
    prefixes = [Fraction(0)]
    for value in means:
        prefixes.append(prefixes[-1] + value - drift)
    if prefixes[-1] != 0:
        raise RuntimeError("the exact centered phase-source cycle did not close")
    return PeriodicPhaseSourceBudget(
        sources,
        duration,
        len(sources),
        means,
        drift,
        tuple(prefixes),
        exact_duration * sum(means, Fraction(0)),
        exact_duration * max(map(abs, prefixes)),
    )


@dataclass(frozen=True, slots=True)
class PeriodicPhaseSourceCompensation:
    """One necessary mean-band check for a declared periodic phase source.

    The reconstructed mean includes supplied accumulated nonphase area.
    Membership of this scalar mean in the band is necessary, not sufficient,
    for all six reconstructed coordinates to remain there. This result does
    not certify the supplied area, phase cycle, event admission or future
    compensation. Visible mean additionally has the carried endpoint term.
    """

    reference: PeriodicPhaseSourceBudget
    block_count: int
    complete_periods: int
    remainder_blocks: int
    initial_mean: Fraction
    actual_nonphase_area: Fraction
    epi_lower: float
    epi_upper: float
    linear_phase_area: Fraction
    periodic_phase_area: Fraction
    phase_area: Fraction
    reconstructed_mean_change: Fraction
    reconstructed_mean: Fraction
    required_compensation_lower: Fraction
    required_compensation_upper: Fraction
    mean_in_band: bool

    @property
    def future_compensation_certified(self) -> bool:
        return False


def observe_periodic_phase_source_compensation(
    reference: PeriodicPhaseSourceBudget,
    *,
    block_count: int,
    initial_mean: Fraction,
    actual_nonphase_area: Fraction,
    epi_lower: float = 0.05,
    epi_upper: float = 1.0,
) -> PeriodicPhaseSourceCompensation:
    """Compute the signed compensation a bounded carried mean would require.

    With unit capacities on C6, let E_N be the accumulated mean area from
    p-A over N blocks, including all refreshed substeps. Exact EPI diffusion
    has zero arithmetic mean, so E_N includes its numerical realization and
    final channel assembly; it is distinct from the weighted phase source.
    The caller supplies that exact accumulated quantity. No producer or
    actual trajectory is inferred from its value.

    The reconstructed mean is m_0+H*(N*b+C_r)+E_N. Necessary mean-band
    membership requires E_N in [ell-m_0-phase_area, U-m_0-phase_area].
    A bounded mean over every N therefore requires E_N/N -> -H*b. A
    nonzero periodic source average cannot be dismissed as a bounded
    periodic offset; when b=0 the nonphase prefix still needs control.
    All reference caches are rebuilt from their raw sources and duration.
    This is not a six-node trapping theorem or a guarantee of future
    phase closure, admission, compensating errors or pressure summability.
    """
    if type(reference) is not PeriodicPhaseSourceBudget:
        raise TypeError("reference must be a PeriodicPhaseSourceBudget")
    rebuilt = derive_periodic_phase_source_budget(
        phase_contributions=reference.phase_contributions,
        block_duration=reference.block_duration,
    )
    if type(block_count) is not int or block_count < 0:
        raise ValueError("block_count must be a nonnegative integer, not a boolean")
    if type(initial_mean) is not Fraction or type(actual_nonphase_area) is not Fraction:
        raise TypeError(
            "initial_mean and actual_nonphase_area must be exact Fraction values"
        )
    lower = _finite_binary64(epi_lower, "epi_lower")
    upper = _finite_binary64(epi_upper, "epi_upper")
    if not 0 < lower <= upper <= 1:
        raise ValueError("the EPI band must satisfy 0 < lower <= upper <= 1")
    exact_lower, exact_upper = Fraction.from_float(lower), Fraction.from_float(upper)
    if not exact_lower <= initial_mean <= exact_upper:
        raise ValueError("initial_mean must lie in the declared EPI band")
    if block_count == 0 and actual_nonphase_area != 0:
        raise ValueError("zero blocks must have zero accumulated nonphase area")
    whole, remainder = divmod(block_count, rebuilt.period)
    duration = Fraction.from_float(rebuilt.block_duration)
    linear = duration * block_count * rebuilt.mean_source
    periodic = duration * rebuilt.centered_prefixes[remainder]
    phase_area = linear + periodic
    change = phase_area + actual_nonphase_area
    final = initial_mean + change
    required_lower = exact_lower - initial_mean - phase_area
    required_upper = exact_upper - initial_mean - phase_area
    in_band = required_lower <= actual_nonphase_area <= required_upper
    if in_band != (exact_lower <= final <= exact_upper):
        raise RuntimeError(
            "the exact source-compensation interval lost its mean identity"
        )
    return PeriodicPhaseSourceCompensation(
        rebuilt,
        block_count,
        whole,
        remainder,
        initial_mean,
        actual_nonphase_area,
        lower,
        upper,
        linear,
        periodic,
        phase_area,
        change,
        final,
        required_lower,
        required_upper,
        in_band,
    )


@dataclass(frozen=True, slots=True)
class NodalAreaCrossing:
    """Exact zero and sign-crossing indices of d+n*a for one coordinate.

    A continuous zero index can be negative or nonintegral. The integer
    zero candidate is retained only when nonnegative, independently of
    the declared prefix. Initially zero coordinates have candidate zero
    and no directional crossing; ``zero_for_all_steps`` distinguishes a
    stationary coordinate. For initially nonzero d, the first crossing
    reaches zero or the opposite sign and can overshoot without any exact
    integer zero. Candidate values beyond the prefix are conditional
    arithmetic predictions, not executed or producer-certified steps.
    """

    initial_area: Fraction
    increment: Fraction
    continuous_zero_step: Fraction | None
    integer_zero_step: int | None
    zero_for_all_steps: bool
    exact_zero_in_prefix: bool
    first_zero_or_opposite_step: int | None
    crossing_in_prefix: bool
    area_before_crossing: Fraction | None
    area_at_crossing: Fraction | None
    crossing_is_exact_zero: bool


@dataclass(frozen=True, slots=True)
class NodalAreaCrossings:
    """A supplied affine nodal-area prefix, without pressure or band provenance.

    Every coordinate retains its original accumulated-area baseline.
    ``positive_joint_zero_step`` requires the whole vector to vanish at
    one strictly positive integer, not merely its mean. If the entire
    vector is stationary zero, every index is a zero and the earliest
    positive candidate is one. Otherwise all constrained roots must agree.
    The prefix flag separately checks the declared maximum step count.
    No EPI endpoint, pressure refresh, flow admission or return to a graph
    state is inferred from a supplied area zero.
    """

    initial_area: Vector
    timestep: float
    capacity: tuple[float, ...]
    pressure: tuple[float, ...]
    max_steps: int
    exact_increment: Vector
    endpoint_area: Vector
    coordinates: tuple[NodalAreaCrossing, ...]
    initially_joint_zero: bool
    positive_joint_zero_step: int | None
    joint_zero_in_prefix: bool
    joint_zero_for_all_steps: bool

    @property
    def pressure_provenance_certified(self) -> bool:
        return False

    @property
    def band_provenance_certified(self) -> bool:
        return False


def derive_nodal_area_crossings(
    *,
    initial_area: tuple[Fraction, ...],
    timestep: float,
    capacity: tuple[float, ...],
    pressure: tuple[float, ...],
    max_steps: int,
) -> NodalAreaCrossings:
    """Solve exact scalar crossings and simultaneous vector zeroes algebraically.

    The declared constant increments are a_i=F(h)*F(nu_i)*F(p_i), with
    finite represented h>=0 and nu_i>=0. For a_i!=0 the only continuous
    zero of d_i+n*a_i is -d_i/a_i. It is an integer zero precisely when
    that ratio is a nonnegative integer. When d_i and a_i have opposite
    signs, ceil(-d_i/a_i) is the first nonnegative integer step reaching
    zero or crossing its sign; equality and overshoot remain separate.

    The initial accumulated areas are arbitrary exact Fractions. This
    algebra neither modifies them to improve balance nor infers where
    they came from. Initially zero coordinates are reported explicitly,
    not counted as a later repayment. Simultaneous positive-integer zero
    requires compatible roots in all coordinates; a nonzero constant
    coordinate prevents it and a constant zero imposes no restriction.

    All candidate indices are retained independently of max_steps and
    accompanied by prefix-membership flags. A cell-horizon or trajectory
    owner must establish that the same input remains valid for that many
    transitions. In particular an old pressure can drive its cell-exit
    transition but cannot be assumed valid after the subsequent refresh.
    This function performs no numerical evolution or pressure production.
    """
    if type(initial_area) is not tuple:
        raise TypeError("initial_area must be an ordered tuple of exact Fractions")
    if not initial_area:
        raise ValueError("initial_area must be nonempty")
    if any(type(value) is not Fraction for value in initial_area):
        raise TypeError("every initial area must be an exact Fraction")
    h = _finite_binary64(timestep, "timestep")
    capacities = _binary64_tuple(capacity, "capacity")
    pressures = _binary64_tuple(pressure, "pressure")
    if len(capacities) != len(initial_area) or len(pressures) != len(initial_area):
        raise ValueError("capacity and pressure must match the initial-area dimension")
    if h < 0 or any(value < 0 for value in capacities):
        raise ValueError("timestep and capacity must be nonnegative")
    if type(max_steps) is not int:
        raise TypeError("max_steps must be a nonnegative integer")
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    increments = tuple(
        Fraction(h) * Fraction(nu) * Fraction(p)
        for nu, p in zip(capacities, pressures, strict=True)
    )
    coordinates = []
    for d, a in zip(initial_area, increments, strict=True):
        root = -d / a if a else None
        zero = None
        if d == 0:
            zero = 0
        elif root is not None and root >= 0 and root.denominator == 1:
            zero = root.numerator
        stationary = d == 0 and a == 0
        crossing = before = after = None
        if d * a < 0:
            crossing = -((-root.numerator) // root.denominator)
            before, after = d + (crossing - 1) * a, d + crossing * a
        coordinates.append(
            NodalAreaCrossing(
                d,
                a,
                root,
                zero,
                stationary,
                stationary or zero is not None and zero <= max_steps,
                crossing,
                crossing is not None and crossing <= max_steps,
                before,
                after,
                crossing is not None and after == 0,
            )
        )
    initially_zero = not any(initial_area)
    stationary_vector = initially_zero and not any(increments)
    joint = None
    if stationary_vector:
        joint = 1
    elif not any(
        d != 0 and a == 0 for d, a in zip(initial_area, increments, strict=True)
    ):
        roots = {
            coordinate.continuous_zero_step
            for coordinate in coordinates
            if coordinate.increment
        }
        if len(roots) == 1:
            candidate = next(iter(roots))
            if candidate > 0 and candidate.denominator == 1:
                joint = candidate.numerator
    return NodalAreaCrossings(
        initial_area,
        h,
        capacities,
        pressures,
        max_steps,
        increments,
        tuple(d + max_steps * a for d, a in zip(initial_area, increments, strict=True)),
        tuple(coordinates),
        initially_zero,
        joint,
        joint is not None and joint <= max_steps,
        stationary_vector,
    )


@dataclass(frozen=True, slots=True)
class TwoLevelNodalReturn:
    """Necessary primitive counts for exact return of a two-level coordinate.

    The common represented duration and positive capacity are fixed, so
    their factors cancel from the exact nodal-area equation. A nonempty
    zero-area word using only these two pressure levels must repeat the
    primitive negative/positive counts an integer number of times. This
    scalar necessity supplies no feasible ordering, carried itinerary,
    simultaneous return of the other coordinates or invariant class.
    """

    negative_pressure: float
    positive_pressure: float
    common_denominator: int
    negative_integer: int
    positive_integer: int
    gcd: int
    minimum_negative_steps: int
    minimum_positive_steps: int
    minimum_total_steps: int

    @property
    def pressure_provenance_certified(self) -> bool:
        return False

    @property
    def periodic_execution_certified(self) -> bool:
        return False


def derive_two_level_nodal_return(
    *,
    negative_pressure: float,
    positive_pressure: float,
) -> TwoLevelNodalReturn:
    """Derive exact count divisibility for two opposite represented sources.

    Put p_minus=-M/D and p_plus=N/D on their least common exact dyadic
    denominator, with positive integers M,N. Under a common fixed positive
    timestep and capacity, a coordinate's zero accumulated nodal area
    requires n_minus*M=n_plus*N. Writing g=gcd(M,N), the primitive positive
    counts are (N/g,M/g); every nonempty solution is a positive integer
    multiple and has length divisible by (M+N)/g.

    This statement begins and ends with supplied pressure levels. It
    does not establish their producer, persistence, ordering or the
    availability of any incoming carry. Additional pressure levels or
    changing timestep/capacity invalidate the two-level count inference.
    A previously accumulated nonzero area requires its own affine budget.
    There is no numerical evolution or change to a physical coefficient.
    """
    negative = _finite_binary64(negative_pressure, "negative_pressure")
    positive = _finite_binary64(positive_pressure, "positive_pressure")
    if not negative < 0 < positive:
        raise ValueError(
            "the two pressure levels must have strictly opposite declared signs"
        )
    _, denominator, integers = _integerized_pressure_levels((negative, positive))
    negative_integer, positive_integer = -integers[0], integers[1]
    divisor = gcd(negative_integer, positive_integer)
    negative_count, positive_count = (
        positive_integer // divisor,
        negative_integer // divisor,
    )
    return TwoLevelNodalReturn(
        negative,
        positive,
        denominator,
        negative_integer,
        positive_integer,
        divisor,
        negative_count,
        positive_count,
        negative_count + positive_count,
    )


def _integerized_pressure_levels(pressure_levels):
    """Retain supplied binary64 levels on their least common exact denominator."""
    levels = _binary64_tuple(pressure_levels, "pressure_levels")
    exact = tuple(Fraction(value) for value in levels)
    denominator = lcm(*(value.denominator for value in exact))
    integers = tuple(
        value.numerator * (denominator // value.denominator) for value in exact
    )
    return levels, denominator, integers


@dataclass(frozen=True, slots=True)
class FiniteLevelNodalReturn:
    """Necessary scalar return-length divisibility for finitely many sources.

    ``arithmetic_zero_area_possible`` means that some nonempty multiset of
    the supplied scalar levels, with repetitions allowed, sums to zero.
    It holds exactly when a zero level or both signs are present. It does
    not establish a pressure-producing state, ordering, compatible carry,
    full-vector return or graph execution.

    ``necessary_length_multiple`` is a congruence condition, not the
    shortest possible return or a sufficient condition at that length.
    A factor of one imposes no congruence restriction. With unequal
    one-sign levels a factor is still reported, but arithmetic return is
    independently impossible. Equal nonzero levels have no valid return
    length and use None. A selectable zero level permits every positive
    length by repeating zero and always gives factor one.
    """

    pressure_levels: tuple[float, ...]
    common_denominator: int
    integer_levels: tuple[int, ...]
    reference_integer: int
    difference_gcd: int
    residue_gcd: int
    necessary_length_multiple: int | None
    has_negative: bool
    has_zero: bool
    has_positive: bool
    arithmetic_zero_area_possible: bool

    @property
    def pressure_provenance_certified(self) -> bool:
        return False

    @property
    def periodic_execution_certified(self) -> bool:
        return False


def derive_finite_level_nodal_return(
    *,
    pressure_levels: tuple[float, ...],
) -> FiniteLevelNodalReturn:
    """Derive a necessary return-length divisor without evolving nodal state.

    A common fixed positive duration and capacity cancel from the exact
    scalar nodal-area equation. Put all supplied binary64 pressures on
    their least common dyadic denominator Q, giving integers z_j. With
    z_0 as an anchor and d=gcd_j(z_j-z_0), any word of length T has integer
    sum congruent to T*z_0 modulo d. Therefore zero accumulated area
    requires d to divide T*z_0. For d>0 this implies that T is a multiple
    of d/gcd(d,z_0). All arithmetic is exact; no levels are adjusted.

    If d=0 every level is equal: zero admits every length, while a nonzero
    value never sums to zero in a nonempty word. For d>0, zero presence or
    opposite signs gives an unconstrained zero-sum multiset; opposite
    rational signs admit the primitive two-level count construction.
    Neither this existence fact nor the length congruence supplies a
    realizable ordering or guarantees a return at any particular length.

    Reordering and duplicate levels do not change the modulus. Adding
    levels can only weaken it: whenever the earlier modulus is defined,
    the new one divides it. Indeed after a common denominator refinement
    by t, the new difference gcd divides t*d, while the anchor becomes
    t*z_0; prime-by-prime cancellation gives the divisibility relation.
    These are scalar arithmetic statements, not future invariance proofs.
    Additional levels must be included before applying the certificate;
    changing duration or capacity also changes the return equation.
    """
    levels, denominator, integers = _integerized_pressure_levels(pressure_levels)
    anchor = integers[0]
    difference_divisor = gcd(*(value - anchor for value in integers))
    residue_divisor = gcd(difference_divisor, anchor)
    if difference_divisor:
        length_multiple = difference_divisor // residue_divisor
    else:
        length_multiple = 1 if anchor == 0 else None
    negative = any(value < 0 for value in integers)
    zero = any(value == 0 for value in integers)
    positive = any(value > 0 for value in integers)
    return FiniteLevelNodalReturn(
        levels,
        denominator,
        integers,
        anchor,
        difference_divisor,
        residue_divisor,
        length_multiple,
        negative,
        zero,
        positive,
        zero or negative and positive,
    )
