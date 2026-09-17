"""A sufficient binary64 pressure-equilibrium obstruction on a unit C6.

This is a detached numerical-kernel theorem with fixed represented phases,
unit capacities, fixed coefficients and unit cycle support. It recomputes
the weighted phase contribution through the shared CPU pressure kernel.
It neither evolves a graph nor asserts that a canonical event keeps those
phases fixed. In particular it proves neither band exit nor mean drift.
"""

from dataclasses import dataclass
from fractions import Fraction
import math

from .._binary64 import uses_ieee_binary64_rounding
from ..dynamics import fused_dnfr
from ..dynamics._euler_kernel import _binary64_tuple, _finite_binary64
from ..mathematics._phase_midpoint import CertifiedTwoNeighborPhase, certified_two_neighbor_phase
from .binary64_nodal_flow import Binary64AdditionCell, _rounding_cell

__all__ = [
    "Binary64PressureEquilibriumRow", "Binary64C6PressureEquilibriumObstruction",
    "derive_binary64_c6_pressure_equilibrium_obstruction",
]


@dataclass(frozen=True, slots=True)
class Binary64PressureEquilibriumRow:
    """One necessary cancellation condition on an unbounded gradient lattice.

    The inverse interval is the rounding cell of -A divided by the positive
    EPI coefficient. Grid indices refer to multiples of 2^-58. A nonempty
    interval does not certify a gradient attainable by any bounded EPI tuple,
    much less simultaneous compatibility of all six cycle rows.
    """

    index: int
    phase_response: CertifiedTwoNeighborPhase
    phase_gradient: float
    phase_contribution: float
    target_epi_pressure: float
    cancellation_cell: Binary64AdditionCell
    inverse_gradient_lower: Fraction
    inverse_gradient_upper: Fraction
    inverse_gradient_lower_closed: bool
    inverse_gradient_upper_closed: bool
    first_grid_index: int
    last_grid_index: int
    grid_excluded: bool


@dataclass(frozen=True, slots=True)
class Binary64C6PressureEquilibriumObstruction:
    """Sufficient fixed-phase obstruction; no root-existence converse.

    ``no_zero_pressure`` is true when at least one row has no lattice
    preimage for cancellation. False means this observer is inconclusive.
    The supplied phases and coefficients are numerical inputs, not causal
    evidence that a graph generated them or will retain them.
    """

    phase: tuple[float, ...]
    epi_weight: Fraction
    phase_weight: Fraction
    epi_lower: float
    epi_upper: float
    gradient_quantum: Fraction
    rows: tuple[Binary64PressureEquilibriumRow, ...]
    no_zero_pressure: bool

    @property
    def fixed_positive_step_convergence_excluded(self) -> bool:
        """Conditional exclusion for carried X with fixed phases and h>0.

        If an excluded row exists, its represented pressure is nonzero at
        every EPI tuple in the band. With unit capacity, each exact carried
        increment has magnitude at least h*2^-1074. Hence increments cannot
        tend to zero under a fixed positive step (or a positive step lower
        bound), and reconstructed X cannot converge while remaining in the
        declared domain. Bounded oscillation is not excluded. Changing
        phases, vanishing steps or leaving the domain invalidates this
        inference; full UM/IL execution is outside it.
        """
        return self.no_zero_pressure

    @property
    def positive_band_exit_certified(self) -> bool:
        return False


def derive_binary64_c6_pressure_equilibrium_obstruction(
    *, phase: tuple[float, ...], epi_weight: float, phase_weight: float,
    epi_lower: float = .05, epi_upper: float = 1.0,
) -> Binary64C6PressureEquilibriumObstruction:
    """Derive necessary row cancellation cells for the actual CPU assembly.

    The declared band must be a positive subinterval of [.05,1]. Every
    binary64 EPI in that range is a multiple of 2^-57. On unit C6 each
    linear row has two unit weights, hence probabilities exactly 1/2.
    Rounded subtraction preserves the 2^-57 lattice: an exact grid point
    is either representable or rounds on a coarser binary lattice. Halving
    is exact here; rounding the two-term sum preserves the 2^-58 lattice.
    Thus the ordinary scalar/vector rows return RN(w_epi*q), q in 2^-58 Z.
    The mixed-sign rational fallback computes the exact neighbor average
    minus center on the same lattice and rounds the final product only.
    Nonzero differences in this band are normal, so no range-loss branch
    can introduce finer gradients. This describes an image inclusion,
    not surjectivity of the lattice onto bounded EPI configurations.

    Fixed unit capacity and degree two make frequency/topology gradients
    zero. For the six certified phase rows the CPU pressure assembly is
    p_i=RN(A_i+RN(w_epi*q_i)), where A_i=RN(w_phase*RN(delta_i/math.pi))
    and delta_i is the shared correctly rounded true-circle midpoint
    displacement. A_i is recomputed through that shared fused kernel,
    using a uniform zero-EPI numerical probe, not a graph preparation.

    Both operands of the final addition are binary64 multiples of 2^-1074;
    their sum can round to zero only when it is exactly zero. Therefore
    p_i=0 requires RN(w_epi*q_i)=-A_i. Invert the nearest-even cell of
    -A_i, preserving its open/closed ties, and test its intersection with
    the unbounded gradient lattice. One empty row proves absence of any
    zero-pressure EPI tuple in the band. Nonempty rows remain inconclusive.
    No coefficients or source means are adjusted or projected.
    """
    phases = _binary64_tuple(phase, "phase")
    if len(phases) != 6:
        raise ValueError("the phase tuple must contain the six ordered C6 vertices")
    lower = _finite_binary64(epi_lower, "epi_lower")
    upper = _finite_binary64(epi_upper, "epi_upper")
    if not .05 <= lower <= upper <= 1.0:
        raise ValueError("the declared EPI band must be a subinterval of [.05,1]")
    e = _finite_binary64(epi_weight, "epi_weight")
    a = _finite_binary64(phase_weight, "phase_weight")
    if not 0.0 < e <= 1.0 or not 0.0 < a <= 1.0:
        raise ValueError("the represented channel weights must lie in (0,1]")
    if not uses_ieee_binary64_rounding():
        raise RuntimeError("the pressure lattice requires the declared IEEE binary64 rounding behavior")
    if fused_dnfr.np is None:
        raise RuntimeError("the pressure lattice requires the shared NumPy CPU pressure kernel")
    responses = tuple(certified_two_neighbor_phase(phases[i], phases[i - 1], phases[(i + 1) % 6])
                      for i in range(6))
    if any(response is None for response in responses):
        raise ValueError("every C6 row must have a certified strict two-neighbor phase midpoint")
    np = fused_dnfr.np
    source = np.asarray(tuple(i for i in range(6) for _ in range(2)), dtype=np.intp)
    target = np.asarray(tuple(j for i in range(6) for j in ((i - 1) % 6, (i + 1) % 6)), dtype=np.intp)
    contributions = fused_dnfr.compute_fused_gradients_symmetric(
        edge_src=source, edge_dst=target, phase=np.asarray(phases, dtype=float),
        epi=np.zeros(6, dtype=float), vf=np.ones(6, dtype=float),
        weights={"w_epi": e, "w_phase": a}, edge_weight=np.ones(12, dtype=float),
        accumulate_both_directions=False, use_jit=False,
    )
    if len(contributions) != 6:
        raise RuntimeError("the shared phase-source probe changed its C6 dimensions")
    quantum = Fraction(1, 2**58)
    exact_weight = Fraction.from_float(e)
    rows = []
    for index, response in enumerate(responses):
        gradient = response.delta / math.pi
        contribution = _finite_binary64(float(contributions[index]), "weighted phase contribution")
        if contribution != a * gradient:
            raise RuntimeError("the shared CPU source differs from the certified phase assembly")
        target_pressure = -contribution
        cell = _rounding_cell(target_pressure, Fraction.from_float(target_pressure))
        inverse_lower, inverse_upper = cell.lower / exact_weight, cell.upper / exact_weight
        scaled_lower, scaled_upper = inverse_lower / quantum, inverse_upper / quantum
        first = -((-scaled_lower.numerator) // scaled_lower.denominator)
        last = scaled_upper.numerator // scaled_upper.denominator
        if not cell.even_significand:
            if scaled_lower.denominator == 1:
                first += 1
            if scaled_upper.denominator == 1:
                last -= 1
        rows.append(Binary64PressureEquilibriumRow(
            index, response, gradient, contribution, target_pressure, cell,
            inverse_lower, inverse_upper, cell.even_significand, cell.even_significand,
            first, last, first > last,
        ))
    result_rows = tuple(rows)
    return Binary64C6PressureEquilibriumObstruction(
        phases, exact_weight, Fraction.from_float(a), lower, upper, quantum,
        result_rows, any(row.grid_excluded for row in result_rows),
    )
