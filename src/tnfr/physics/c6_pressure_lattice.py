"""Exact local C6 pressure, sign-sector and frozen-stencil bounds.

This detached observer binds the existing CPU pressure and linear-reducer
owners on a declared EPI slab, including a Cartesian trapping obstruction.
It does not execute a graph, establish that
the supplied phases are reachable, or exclude correlated invariant sets.
"""

from dataclasses import dataclass
from fractions import Fraction
import math

from ..dynamics import fused_dnfr
from ..dynamics._euler_kernel import (
    NodalRemainderState, _binary64_tuple, _finite_binary64, _validate_nodal_remainder_state,
)
from ..mathematics._neighbor_differences import edge_mean_differences, mean_neighbor_difference
from ._cycle_algebra import Vector, laplacian_action
from .binary64_nodal_flow import Binary64AdditionCell
from .binary64_pressure_equilibrium import (
    Binary64C6PressureEquilibriumObstruction,
    Binary64PressureEquilibriumRow,
    derive_binary64_c6_pressure_equilibrium_obstruction,
)
from .nodal_remainder import derive_nodal_remainder_cell_horizon
from .nodal_remainder_pressure import observe_finite_nodal_pressure_drift

__all__ = [
    "C6PressureLatticeRow", "C6PressureLatticeReference", "C6PressureLatticeObservation",
    "derive_c6_pressure_lattice", "observe_c6_pressure_lattice",
    "C6PressureSignSector", "C6PressureSectorExit",
    "derive_c6_pressure_sign_sector", "observe_c6_pressure_sector_exit",
    "C6FrozenPressureStencil", "observe_c6_frozen_pressure_stencil",
]


@dataclass(frozen=True, slots=True)
class C6PressureLatticeRow:
    """Exact integer thresholds for nonnegative and nonpositive pressure.

    The source's inverse rounding cell retains its endpoint parity. The
    new indices use this reference's local gradient quantum, rather than
    the coarser-domain index convention of the source observer.
    """

    source: Binary64PressureEquilibriumRow
    nonnegative_min_index: int
    nonpositive_max_index: int


@dataclass(frozen=True, slots=True)
class C6PressureLatticeReference:
    """A conditional numerical sign theorem for one fixed phase source.

    A true ``every_pressure_has_positive`` excludes every all-nonpositive
    pressure vector in the slab; the negative flag has the opposite role.
    Either flag excludes forward invariance of any nonempty Cartesian
    product of finite admissible reconstructed-coordinate sets in the
    slab under a positive carried nodal step. Such a product has its
    simultaneous upper and lower corners. At the appropriate corner one
    exact coordinate moves outward, irrespective of its existing carry.

    The same argument applies to closed coordinate intervals whose corner
    states have admissible canonical encodings. It does not exclude a
    correlated trapping set, bounded oscillation or temporal compensation.
    There is no claim of exit from the complete positive EPI band.
    """

    source: Binary64C6PressureEquilibriumObstruction
    sources: tuple[float, ...]
    epi_quantum: Fraction
    gradient_quantum: Fraction
    rows: tuple[C6PressureLatticeRow, ...]
    nonnegative_min_sum: int
    nonpositive_max_sum: int
    every_pressure_has_positive: bool
    every_pressure_has_negative: bool
    no_cartesian_trap: bool

    @property
    def positive_band_exit_certified(self) -> bool:
        return False


def derive_c6_pressure_lattice(
    *, phase: tuple[float, ...], epi_weight: float, phase_weight: float,
    epi_lower: float = .375, epi_upper: float = .625,
) -> C6PressureLatticeReference:
    """Derive joint sign constraints from exact cycle-gradient compatibility.

    Let delta=ulp(epi_lower). The band is a subinterval of [.05,1] with
    width at most 2^52*delta. Every displayed EPI is on the delta lattice.
    The width condition implies upper<=2*lower, so each subtraction is
    exact by Sterbenz. The two half-differences are normal when nonzero;
    their exact sum is (delta/2)*m_i with |m_i|<=2^53 and is representable.
    Both ordinary reducers and the mixed-sign rational fallback therefore
    return RN(w_epi*(delta/2)*m_i), with the same exact integers
    m_i=n_(i-1)+n_(i+1)-2*n_i. In particular sum(m_i)=0.

    Reuse the shared phase-source probe and the exact inverse nearest-even
    cells of -A_i. The lower cell boundary determines the first integer
    whose rounded EPI contribution is >=-A_i; the upper boundary determines
    the last integer whose contribution is <=-A_i. Endpoint parity matters,
    including a zero target or underflowing coefficient product. A sum of
    two finite binary64 operands rounds to zero only when they cancel
    exactly, so these are also the exact pressure-sign thresholds.

    If the sum of nonpositive maxima is negative, the conservation identity
    rules out an all-nonpositive pressure vector. If the sum of nonnegative
    minima is positive, it rules out an all-nonnegative vector. Integer
    ranges are deliberately not clipped to coordinate feasibility: these
    relaxed necessary conditions suffice, without a root-existence converse.

    Default endpoints define an analytic slab crossing the .5 binade; they
    do not alter the pressure coefficients or prepare a dynamical state.
    The theorem concerns unit C6 support/capacity, zero Gamma and the shared
    CPU pressure assembly with zero frequency/topology contributions.
    """
    lower = _finite_binary64(epi_lower, "epi_lower")
    upper = _finite_binary64(epi_upper, "epi_upper")
    if not .05 <= lower <= upper <= 1.0:
        raise ValueError("the EPI slab must be a subinterval of [.05,1]")
    delta = Fraction.from_float(math.ulp(lower))
    if Fraction.from_float(upper) - Fraction.from_float(lower) > 2**52 * delta:
        raise ValueError("the EPI slab width must not exceed 2^52*ulp(epi_lower)")
    source = derive_binary64_c6_pressure_equilibrium_obstruction(
        phase=phase, epi_weight=epi_weight, phase_weight=phase_weight,
        epi_lower=lower, epi_upper=upper,
    )
    quantum = delta / 2
    rows = []
    for row in source.rows:
        lo = row.inverse_gradient_lower / quantum
        hi = row.inverse_gradient_upper / quantum
        first = -((-lo.numerator) // lo.denominator)
        last = hi.numerator // hi.denominator
        if not row.inverse_gradient_lower_closed and lo.denominator == 1:
            first += 1
        if not row.inverse_gradient_upper_closed and hi.denominator == 1:
            last -= 1
        rows.append(C6PressureLatticeRow(row, first, last))
    result_rows = tuple(rows)
    first_sum = sum(row.nonnegative_min_index for row in result_rows)
    last_sum = sum(row.nonpositive_max_index for row in result_rows)
    positive, negative = last_sum < 0, first_sum > 0
    return C6PressureLatticeReference(
        source, tuple(row.phase_contribution for row in source.rows),
        delta, quantum, result_rows, first_sum, last_sum, positive, negative,
        positive or negative,
    )


def _represented_source_weight(value: Fraction, label: str) -> float:
    if type(value) is not Fraction:
        raise TypeError(f"{label} must retain its exact represented Fraction")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite represented weight") from exc
    _finite_binary64(result, label)
    if Fraction.from_float(result) != value:
        raise ValueError(f"{label} must equal an actual binary64 value")
    return result


def _rebuild_lattice(reference: C6PressureLatticeReference) -> C6PressureLatticeReference:
    if not isinstance(reference, C6PressureLatticeReference):
        raise TypeError("reference must be a C6PressureLatticeReference")
    source = reference.source
    if not isinstance(source, Binary64C6PressureEquilibriumObstruction):
        raise TypeError("reference source must be a binary64 C6 pressure source")
    return derive_c6_pressure_lattice(
        phase=source.phase,
        epi_weight=_represented_source_weight(source.epi_weight, "epi_weight"),
        phase_weight=_represented_source_weight(source.phase_weight, "phase_weight"),
        epi_lower=source.epi_lower, epi_upper=source.epi_upper,
    )


@dataclass(frozen=True, slots=True)
class C6PressureLatticeObservation:
    """One supplied EPI tuple, shared-kernel bindings and exact mean budget.

    The EPI reduction error is RN(w_epi*q)-w_epi*q. The assembly error is
    p-A-RN(w_epi*q). Since the exact cycle gradients sum to zero, the mean
    pressure is the phase-source mean plus these two signed error means.
    A zero observed mean establishes no future compensation or trapping.
    """

    reference: C6PressureLatticeReference
    epi: tuple[float, ...]
    epi_indices: tuple[int, ...]
    gradient_indices: tuple[int, ...]
    gradient_index_sum: int
    gradient: Vector
    epi_contributions: tuple[float, ...]
    pressure: tuple[float, ...]
    epi_reduction_error: Vector
    assembly_error: Vector
    mean_phase_contribution: Fraction
    mean_epi_reduction_error: Fraction
    mean_assembly_error: Fraction
    mean_pressure: Fraction
    mean_identity_residual: Fraction
    scalar_reducer_agreement: bool
    vector_reducer_agreement: bool
    fused_pressure_agreement: bool


def observe_c6_pressure_lattice(
    reference: C6PressureLatticeReference, *, epi: tuple[float, ...],
) -> C6PressureLatticeObservation:
    """Bind a static local EPI reading to both reducers and the CPU assembly.

    Rebuild all public reference caches from their primitive source inputs.
    The rebuild includes the shared zero-EPI phase-source probe; one further
    fused call reads the actual supplied EPI tuple. No pressure is projected,
    no carry or graph is changed, and no trajectory is inferred.
    """
    reference = _rebuild_lattice(reference)
    return _observe_rebuilt_c6_pressure_lattice(reference, epi)


def _pressure_at_gradient_index(reference, node, index):
    """Read the proved row formula on an already rebuilt source.

    Integer gradient intervals can overapproximate attainable tuples. This
    algebraic readout neither asserts joint realizability nor evolves EPI.
    Keep the product rounding and source assembly in their canonical order.
    """
    if type(node) is not int or not 0 <= node < 6 or type(index) is not int:
        raise ValueError("a C6 row requires a node index in 0..5 and an integer gradient")
    product = float(reference.source.epi_weight * reference.gradient_quantum * index)
    return _finite_binary64(reference.sources[node] + product, "indexed C6 pressure")


def _observe_rebuilt_c6_pressure_lattice(reference, epi):
    """Read one supplied tuple after the public source has been revalidated."""
    values = _binary64_tuple(epi, "epi")
    if len(values) != 6:
        raise ValueError("epi must contain the six ordered C6 vertices")
    source = reference.source
    if any(not source.epi_lower <= value <= source.epi_upper for value in values):
        raise ValueError("every EPI value must lie in the declared local slab")
    exact_values = tuple(Fraction.from_float(value) for value in values)
    origin = Fraction.from_float(source.epi_lower)
    indices = tuple((value - origin) / reference.epi_quantum for value in exact_values)
    gradients = tuple(-value for value in laplacian_action(exact_values))
    gradient_indices = tuple(value / reference.gradient_quantum for value in gradients)
    if any(value.denominator != 1 for value in indices + gradient_indices):
        raise RuntimeError("the EPI tuple lost its proven local lattice")
    integer_indices = tuple(value.numerator for value in indices)
    integer_gradients = tuple(value.numerator for value in gradient_indices)
    index_sum = sum(integer_gradients)
    if index_sum != 0 or any(abs(value) > 2**53 for value in integer_gradients):
        raise RuntimeError("the cycle gradient lost its exact integer compatibility")
    e = float(source.epi_weight)
    expected_epi = tuple(float(source.epi_weight * value) for value in gradients)
    scalar = tuple(mean_neighbor_difference(
        values[i], (values[i - 1], values[(i + 1) % 6]), coefficient=e,
    ) for i in range(6))
    np = fused_dnfr.np
    edge_src = np.asarray(tuple(i for i in range(6) for _ in range(2)), dtype=np.intp)
    edge_dst = np.asarray(tuple(j for i in range(6) for j in ((i - 1) % 6, (i + 1) % 6)), dtype=np.intp)
    array = np.asarray(values, dtype=float)
    vector = _binary64_tuple(tuple(float(value) for value in edge_mean_differences(
        array, edge_src, edge_dst, coefficient=e,
    )), "vector EPI contribution")
    if scalar != expected_epi or vector != expected_epi:
        raise RuntimeError("a shared EPI reducer differs from the exact local lattice product")
    pressure = _binary64_tuple(tuple(float(value) for value in fused_dnfr.compute_fused_gradients_symmetric(
        edge_src=edge_src, edge_dst=edge_dst, phase=np.asarray(source.phase, dtype=float),
        epi=array, vf=np.ones(6, dtype=float),
        weights={"w_epi": e, "w_phase": float(source.phase_weight)},
        edge_weight=np.ones(12, dtype=float), accumulate_both_directions=False, use_jit=False,
    )), "assembled pressure")
    expected_pressure = tuple(_pressure_at_gradient_index(reference, i, m)
                              for i, m in enumerate(integer_gradients))
    if pressure != expected_pressure:
        raise RuntimeError("the shared pressure assembly differs from its exact source-plus-EPI binding")
    exact_sources = tuple(Fraction.from_float(value) for value in reference.sources)
    exact_epi = tuple(Fraction.from_float(value) for value in expected_epi)
    exact_pressure = tuple(Fraction.from_float(value) for value in pressure)
    reduction_error = tuple(g - source.epi_weight * q for g, q in zip(exact_epi, gradients, strict=True))
    assembly_error = tuple(p - a - g for p, a, g in zip(exact_pressure, exact_sources, exact_epi, strict=True))
    phase_mean = sum(exact_sources, Fraction(0)) / 6
    reduction_mean = sum(reduction_error, Fraction(0)) / 6
    assembly_mean = sum(assembly_error, Fraction(0)) / 6
    pressure_mean = sum(exact_pressure, Fraction(0)) / 6
    residual = pressure_mean - phase_mean - reduction_mean - assembly_mean
    if residual != 0:
        raise RuntimeError("the local pressure mean decomposition lost its exact identity")
    return C6PressureLatticeObservation(
        reference, values, integer_indices, integer_gradients, index_sum,
        gradients, expected_epi, pressure, reduction_error, assembly_error,
        phase_mean, reduction_mean, assembly_mean, pressure_mean, residual,
        True, True, True,
    )


@dataclass(frozen=True, slots=True)
class C6PressureSignSector:
    """A strict pressure-sign sector of one local integer Laplacian row.

    The sector is clipped to the possible gradient-index range of the
    declared slab. The represented pressure at its boundary is a uniform
    signed lower magnitude throughout that sector. An empty sector has no
    pressure margin. No graph, reachable state or future phase rule is
    inferred from the fixed numerical source.
    """

    reference: C6PressureLatticeReference
    node: int
    sign: int
    minimum_gradient_index: int
    maximum_gradient_index: int
    cut_index: int
    sector_min_index: int
    sector_max_index: int
    empty: bool
    bounding_gradient_index: int | None
    pressure_bound: float | None
    signed_pressure_margin: Fraction | None


def _require_c6_node(node):
    if type(node) is not int:
        raise TypeError("node must be an integer C6 index")
    if not 0 <= node < 6:
        raise ValueError("node must be one of the six ordered C6 indices")
    return node


def _derive_rebuilt_sign_sector(reference, node, sign):
    node = _require_c6_node(node)
    if type(sign) is not int:
        raise TypeError("sign must be the integer -1 or +1")
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    source = reference.source
    width = (Fraction(source.epi_upper) - Fraction(source.epi_lower)) / reference.epi_quantum
    if width.denominator != 1:
        raise RuntimeError("the declared slab endpoints lost their integer EPI lattice")
    minimum, maximum = -2 * width.numerator, 2 * width.numerator
    row = reference.rows[node]
    if sign < 0:
        cut = row.nonnegative_min_index - 1
        sector_min, sector_max = minimum, min(cut, maximum)
    else:
        cut = row.nonpositive_max_index + 1
        sector_min, sector_max = max(cut, minimum), maximum
    empty = sector_min > sector_max
    boundary = pressure = margin = None
    if not empty:
        boundary = sector_max if sign < 0 else sector_min
        weighted_gradient = float(source.epi_weight * reference.gradient_quantum * boundary)
        pressure = _finite_binary64(reference.sources[node] + weighted_gradient, "sector pressure bound")
        margin = sign * Fraction(pressure)
        if margin <= 0:
            raise RuntimeError("the strict pressure sector lost its positive signed margin")
    return C6PressureSignSector(
        reference, node, sign, minimum, maximum, cut, sector_min, sector_max,
        empty, boundary, pressure, margin,
    )


def derive_c6_pressure_sign_sector(
    reference: C6PressureLatticeReference, *, node: int, sign: int,
) -> C6PressureSignSector:
    """Derive a uniform signed pressure bound over a local integer sector.

    Let l be the first nonnegative-pressure index and u the last
    nonpositive-pressure index from the exact inverse rounding cells.
    Negative pressure is equivalent to m<=l-1; positive pressure is
    equivalent to m>=u+1. If l<=u, the intervening integers have zero
    pressure; exiting one strict sector need not enter the opposite one.

    The pressure law p(m)=RN(A+RN(w_epi*Q*m)) is monotone in integer m.
    On a slab of width D EPI lattice units, |m|<=2D. Clip the sign sector
    to [-2D,2D] before evaluating its boundary, which handles small channel
    weights and unreachable cuts without inventing possible gradients.
    For a nonempty sector its nearest-to-zero boundary supplies c>0 with
    sign*p(m)>=c throughout the sector. The coefficient product is formed
    exactly then rounded once, matching both proven linear reducers.
    All public reference caches are rebuilt from primitive source inputs.
    """
    return _derive_rebuilt_sign_sector(_rebuild_lattice(reference), node, sign)


@dataclass(frozen=True, slots=True)
class C6PressureSectorExit:
    """Conditional finite residence bound in a strict nodal sign sector.

    For an initial state outside the sector, or an empty sector, no step
    budget is returned. Otherwise each sector-sourced unit-capacity step
    advances the selected exact coordinate outward by at least h*c.
    ``max_sector_steps`` bounds consecutive such transitions whose exact
    endpoints all remain in the declared band. By ``first_exit_bound``
    the source sector or one of the fixed-source/band/update premises must
    have failed. This disjunction proves neither an opposite sign hit nor
    exit from the complete application band. A zero-pressure sector can
    intervene for sources whose inverse cancellation cells are nonempty.
    """

    sector: C6PressureSignSector
    state: NodalRemainderState
    timestep: float
    initial_observation: C6PressureLatticeObservation
    initial_in_sector: bool
    outward_distance: Fraction
    per_step_margin: Fraction | None
    max_sector_steps: int | None
    first_exit_bound: int | None

    @property
    def opposite_sign_hit_certified(self) -> bool:
        return False

    @property
    def positive_band_exit_certified(self) -> bool:
        return False


def observe_c6_pressure_sector_exit(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    node: int, sign: int, timestep: float,
) -> C6PressureSectorExit:
    """Bound strict-sign residence from a validated incoming carried state.

    Fixed unit capacity, zero Gamma, fixed phase/support/weights and a
    fixed represented h>0 are conditional future premises. Pressure must
    be refreshed from the same shared CPU map on the displayed EPI;
    there are no intervening events, carry resets or EPI projections.
    The incoming numerical state's band must lie inside the reference's
    local slab. Both its displayed EPI and reconstructed coordinates are
    validated, and its present pressure is independently read through the
    shared local observer. No later pressure or trajectory is executed.

    If the selected initial index is in the strict sector, define d as
    X_i-lower for negative pressure or upper-X_i for positive pressure.
    Every valid prefix staying in the sector obeys n*h*c<=d. Its length
    is therefore at most floor(d/(h*c)); the next step cannot
    satisfy all of the same premises. The distance uses the actual carry.
    """
    reference = _rebuild_lattice(reference)
    sector = _derive_rebuilt_sign_sector(reference, node, sign)
    exact = _validate_nodal_remainder_state(state)
    h = _finite_binary64(timestep, "timestep")
    if h <= 0:
        raise ValueError("the sector residence bound requires a strictly positive timestep")
    source = reference.source
    if not source.epi_lower <= state.epi_lower <= state.epi_upper <= source.epi_upper:
        raise ValueError("the numerical state band must be contained in the reference slab")
    observation = _observe_rebuilt_c6_pressure_lattice(reference, state.epi)
    initial_index = observation.gradient_indices[node]
    inside = not sector.empty and sector.sector_min_index <= initial_index <= sector.sector_max_index
    distance = (exact[node] - Fraction(state.epi_lower) if sign < 0
                else Fraction(state.epi_upper) - exact[node])
    margin = maximum = bound = None
    if inside:
        if sign * Fraction(observation.pressure[node]) < sector.signed_pressure_margin:
            raise RuntimeError("the actual initial pressure violates its strict-sector lower bound")
        margin = Fraction(h) * sector.signed_pressure_margin
        ratio = distance / margin
        maximum = ratio.numerator // ratio.denominator
        bound = maximum + 1
    return C6PressureSectorExit(
        sector, state, h, observation, inside, distance, margin, maximum, bound,
    )


@dataclass(frozen=True, slots=True)
class C6FrozenPressureStencil:
    """Conditional residence bounds for one fixed three-node visible stencil.

    The current row is bound to the shared CPU pressure producer. Under
    fixed phases, unit capacities, unit C6 support, channel coefficients
    and zero Gamma, its pressure stays identical while the displayed EPI
    at the previous, center and next node remains fixed. Remote EPI and
    all carries may change without altering this row's pressure.

    The initial bound retains the selected center's actual carry and
    nearest-even ties. The uniform bound covers every admissible incoming
    center carry through its closed cell enclosure, conservatively at odd
    ties. Both concern the center only: a neighbor may change earlier and
    invalidate the frozen stencil before either bound is reached. Failure
    can also be an exact-band or update-contract exit. No sign hit follows.
    Zero pressure supplies no finite center bound or general invariance
    conclusion. These are detached numerical statements, not graph seals.
    """

    reference: C6PressureLatticeReference
    state: NodalRemainderState
    node: int
    timestep: float
    stencil: tuple[int, int, int]
    stencil_epi: tuple[float, float, float]
    gradient_index: int
    pressure: float
    exact_increment: Fraction
    initial_observation: C6PressureLatticeObservation
    center_cell: Binary64AdditionCell
    center_cell_width: Fraction
    initial_max_frozen_steps: int | None
    initial_first_exit_bound: int | None
    uniform_max_frozen_steps: int | None
    uniform_first_exit_bound: int | None

    @property
    def graph_provenance_certified(self) -> bool:
        return False

    @property
    def sign_hit_certified(self) -> bool:
        return False

    @property
    def positive_band_exit_certified(self) -> bool:
        return False


def observe_c6_frozen_pressure_stencil(
    reference: C6PressureLatticeReference, *, state: NodalRemainderState,
    node: int, timestep: float,
) -> C6FrozenPressureStencil:
    """Bind one local pressure row and bound its frozen-stencil residence.

    Both shared linear reducers read only the two neighbors and center
    in this row; the fixed phase contribution uses the same neighbors.
    Unit capacity and degree two give zero frequency/topology gradients.
    Thus with stencil (i-1,i,i+1) fixed, the actual represented row remains
    p_i=RN(A_i+RN(w_epi*Q*m_i)), independently of remote displayed EPI and
    all carried remainders. This conditional locality assertion requires
    unchanged phases/support/coefficients and no jumps or projections.

    Retained carry gives X_i(n)=X_i(0)+n*h*p_i for every valid frozen
    prefix. The existing held-cell owner computes the selected coordinate's
    exact directional limit inside C(x_i) intersect [lower,upper]. Its
    global first-exit step is intentionally not used: other coordinates
    may change while this three-node stencil and its pressure stay fixed.

    The existing finite-drift owner on the one-dimensional center
    projection supplies width W and floor(W/(h*abs(p_i))) as a uniform
    upper bound over incoming carries when p_i!=0. It uses closed cell
    envelopes at odd ties; the actual-carry bound retains exact openness.
    ``center_cell`` is the raw rounding cell, while ``center_cell_width``
    is its band-clipped enclosure width. At zero pressure both owners
    report no finite bound. No hypothetical trajectory is executed.
    """
    reference = _rebuild_lattice(reference)
    node = _require_c6_node(node)
    _validate_nodal_remainder_state(state)
    h = _finite_binary64(timestep, "timestep")
    if h <= 0:
        raise ValueError("a frozen-stencil residence bound requires a strictly positive timestep")
    source = reference.source
    if not source.epi_lower <= state.epi_lower <= state.epi_upper <= source.epi_upper:
        raise ValueError("the numerical state band must be contained in the reference slab")
    observation = _observe_rebuilt_c6_pressure_lattice(reference, state.epi)
    pressure = observation.pressure[node]
    horizon = derive_nodal_remainder_cell_horizon(
        state=state, timestep=h, capacity=(1.,) * 6, pressure=observation.pressure,
    )
    initial_maximum = horizon.coordinate_step_limits[node]
    projected = observe_finite_nodal_pressure_drift(
        epi_states=((state.epi[node],),), pressure_vectors=((pressure,),),
        functional=(Fraction(-1 if pressure < 0 else 1),), timestep=h,
        epi_lower=state.epi_lower, epi_upper=state.epi_upper,
    )
    if initial_maximum is not None and initial_maximum > projected.max_confined_steps:
        raise RuntimeError("the actual-carry stencil bound exceeds its uniform cell-width enclosure")
    stencil = ((node - 1) % 6, node, (node + 1) % 6)
    return C6FrozenPressureStencil(
        reference, state, node, h, stencil, tuple(state.epi[index] for index in stencil),
        observation.gradient_indices[node], pressure, horizon.exact_increment[node],
        observation, horizon.source_cells[node], projected.width, initial_maximum,
        initial_maximum + 1 if initial_maximum is not None else None,
        projected.max_confined_steps, projected.escape_step_bound,
    )
