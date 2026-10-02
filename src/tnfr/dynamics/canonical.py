"""Canonical TNFR nodal equation implementation.

This module provides the explicit, canonical implementation of the fundamental
TNFR nodal equation as specified in the theory:

    ∂EPI/∂t = νf · ΔNFR(t)

These functions evaluate the equation in the engine's signed real scalar EPI
chart. A scalar product does not by itself define the full structural manifold
or identify EPI with a measured physical quantity.

Where:
  - EPI: Primary Information Structure (coherent form)
  - νf: Structural frequency in Hz_str (structural hertz)
  - ΔNFR: Nodal gradient (reorganization operator)
  - t: Declared evolution coordinate; a physical-clock bridge is separate

This implementation supplies numerical evaluation of the stated row by:
  1. Making the canonical equation explicit in code
  2. Checking finite inputs/output and nonnegative structural capacity
  3. Providing clear mapping between theory and implementation
  4. Maintaining reproducibility and traceability

TNFR Invariants (from AGENTS.md):
  - EPI as coherent form: named operators and declared nodal solver steps
  - Structural units: νf expressed in Hz_str (structural hertz)
  - ΔNFR semantics: sign and magnitude modulate reorganization rate
  - Operator closure requires the separate operator and execution contracts

The optional phase/pressure extension below is a configured completion of the
EPI equation, not a derivation of the remaining nodal channel laws. Numeric
validation does not establish units calibration or preservation of invariants.

References:
  - theory/FUNDAMENTAL_THEORY.md: Scalar chart and nodal equation scope
  - AGENTS.md: Foundations and canonical invariants
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import TYPE_CHECKING, Any, NamedTuple

from .._exact_time import finite_represented_real
from ..alias import get_attr, set_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..errors.contextual import FrequencyError, NetworkConfigError, TNFRValueError
from ..types import require_finite_real_scalar_epi

if TYPE_CHECKING:
    from ..types import GraphLike

__all__ = (
    "NodalEquationResult",
    "compute_canonical_nodal_derivative",
    "validate_structural_frequency",
    "validate_nodal_gradient",
    # Extended dynamics with flux fields
    "ExtendedNodalEquationResult",
    "compute_extended_nodal_system",
)


class NodalEquationResult(NamedTuple):
    """Result of canonical nodal equation evaluation.

    Attributes:
        derivative: ∂EPI/∂t computed from νf · ΔNFR(t)
        nu_f: Structural frequency (Hz_str) used in computation
        delta_nfr: Nodal gradient (ΔNFR) used in computation
        validated: Whether finite-input/output and capacity-sign checks ran
    """

    derivative: float
    nu_f: float
    delta_nfr: float
    validated: bool


def compute_canonical_nodal_derivative(
    nu_f: float,
    delta_nfr: float,
    *,
    validate_units: bool = True,
    graph: GraphLike | None = None,
) -> NodalEquationResult:
    """Compute ∂EPI/∂t using the canonical TNFR nodal equation.

    This is the explicit implementation of the fundamental equation:
        ∂EPI/∂t = νf · ΔNFR(t)

    The function computes the time derivative of the Primary Information
    Structure (EPI) as the product of:
      - νf: structural frequency (reorganization rate in Hz_str)
      - ΔNFR: nodal gradient (reorganization need/operator)

    Args:
        nu_f: Structural frequency in Hz_str (must be non-negative)
        delta_nfr: Nodal gradient (reorganization operator)
        validate_units: If True, checks finite inputs/output and nonnegative capacity
        graph: Optional graph for context-aware validation

    Returns:
        NodalEquationResult containing the computed derivative and metadata

    Raises:
        FrequencyError: If validation is enabled and capacity is invalid
        NetworkConfigError: If validation is enabled and pressure or the product is invalid
        TNFRValueError: If validate_units is not a boolean

    Notes:
        - This function is the canonical reference implementation
        - The result represents the instantaneous rate of EPI evolution
        - Units: [∂EPI/∂t] = [EPI]/[t]. When [νf] = 1/[t], pressure
          has the EPI chart's units. The derivative has units Hz_str only
          for a dimensionless EPI chart with structural-time units.
        - Computing the product does not establish physical unit calibration,
          operator closure, or the remaining channel evolution laws.

    Examples:
        >>> # Basic computation
        >>> result = compute_canonical_nodal_derivative(1.0, 0.5)
        >>> result.derivative
        0.5

        >>> # With explicit validation
        >>> result = compute_canonical_nodal_derivative(
        ...     nu_f=1.2,
        ...     delta_nfr=-0.3,
        ...     validate_units=True
        ... )
        >>> result.validated
        True
    """
    if not isinstance(validate_units, bool):
        raise TNFRValueError("validate_units must be a boolean")
    validated = False

    if validate_units:
        nu_f = validate_structural_frequency(nu_f, graph=graph)
        delta_nfr = validate_nodal_gradient(delta_nfr, graph=graph)
        validated = True

    # Canonical TNFR nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    derivative = float(nu_f) * float(delta_nfr)
    if validate_units and not math.isfinite(derivative):
        raise NetworkConfigError(
            parameter="dEPI_dt",
            value=derivative,
            reason="The nodal product must be finite in the represented scalar chart",
        )

    return NodalEquationResult(
        derivative=derivative,
        nu_f=nu_f,
        delta_nfr=delta_nfr,
        validated=validated,
    )


def validate_structural_frequency(
    nu_f: float,
    *,
    graph: GraphLike | None = None,
) -> float:
    """Validate nonnegative represented structural capacity.

    Structural frequency (νf) must satisfy TNFR constraints:
      - Non-negative (νf ≥ 0)
      - Finite real scalar, excluding boolean/text coercions
      - Nonzero input must not underflow to represented zero

    Args:
        nu_f: Structural frequency to validate
        graph: Reserved graph context; does not select numerical bounds

    Returns:
        Validated structural frequency value

    Raises:
        FrequencyError: If capacity fails real-scalar, representation or sign admission

    Notes:
        - νf = 0 is valid and suppresses the unforced nodal product.
        - No finite upper cutoff is imposed; products still require validation.
        - Numeric admission does not calibrate structural or physical units.
    """
    try:
        value, _ = finite_represented_real(nu_f, "nu_f")
    except (TypeError, ValueError) as exc:
        raise FrequencyError(vf=nu_f, operation="validation") from exc

    if value < 0:
        raise FrequencyError(vf=value, operation="validation")

    return value


def validate_nodal_gradient(
    delta_nfr: float,
    *,
    graph: GraphLike | None = None,
) -> float:
    """Validate that nodal gradient is well-defined.

    The nodal gradient (ΔNFR) represents the internal reorganization
    operator and must be:
      - Finite and well-defined
      - Sign indicates reorganization direction
      - Magnitude indicates reorganization intensity

    Args:
        delta_nfr: Nodal gradient to validate
        graph: Optional graph for context-aware validation

    Returns:
        Validated nodal gradient value

    Raises:
        NetworkConfigError: If pressure fails real-scalar or representation admission

    Notes:
        - With positive capacity, the sign selects the positive or negative
          direction of the declared EPI chart; it does not identify the named
          Expansion or Contraction operator.
        - ΔNFR = 0 gives zero EPI rate, not necessarily stationary capacity,
          phase, support or the full structural tetrad.
        - Do NOT reinterpret as classical "error gradient"
        - This scalar is an evaluated structural response. The constitutive
          map producing it is a separate object, not an optimization target.
    """
    return _validate_real_parameter(delta_nfr, "delta_nfr")


def _validate_real_parameter(value: Any, parameter: str) -> float:
    """Apply shared real-scalar admission with this API's contextual error."""
    try:
        return finite_represented_real(value, parameter)[0]
    except (TypeError, ValueError) as exc:
        raise NetworkConfigError(
            parameter=parameter, value=value, reason=str(exc)
        ) from exc


# Extended TNFR dynamics with canonical flux fields
class ExtendedNodalEquationResult(NamedTuple):
    """Result of the optional configured EPI/phase/pressure system.

    Represents the coupled system:
    1. ∂EPI/∂t = νf · ΔNFR(t)           [Classical nodal equation]
    2. ∂θ/∂t = f(νf, ΔNFR, J_φ)        [Phase evolution with transport]
    3. ∂ΔNFR/∂t = g(∇·J_ΔNFR)          [Configured divergence response]

    The phase and pressure equations add operational premises. They are not
    consequences of the EPI equation and supply no capacity evolution law.

    Attributes:
        classical_derivative: ∂EPI/∂t (original TNFR nodal equation)
        phase_derivative: ∂θ/∂t (phase evolution with J_φ transport)
        dnfr_derivative: ∂ΔNFR/∂t (configured divergence response)
        j_phi: Phase current J_φ used in computation
        j_dnfr_divergence: ∇·J_ΔNFR divergence used
        coupling_strength: Local network coupling coefficient
        validated: Whether numeric input and finite-output checks passed
    """

    classical_derivative: float  # ∂EPI/∂t = νf·ΔNFR
    phase_derivative: float  # ∂θ/∂t with J_φ transport
    dnfr_derivative: float  # ∂ΔNFR/∂t from the configured divergence response
    j_phi: float  # Phase current J_φ
    j_dnfr_divergence: float  # Flux divergence ∇·J_ΔNFR
    coupling_strength: float  # Local coupling coefficient
    validated: bool  # Extended validation status


def compute_extended_nodal_system(
    nu_f: float,
    delta_nfr: float,
    theta: float,
    j_phi: float,
    j_dnfr_divergence: float,
    coupling_strength: float = 1.0,
    *,
    validate_units: bool = True,
    graph: GraphLike | None = None,
) -> ExtendedNodalEquationResult:
    """Evaluate an optional operational completion using supplied flux fields.

    The canonical EPI product is retained. The added phase/pressure laws and
    their coefficients are constitutive choices, not deductions from that
    product or from the definitions of J_φ and J_ΔNFR. Structural capacity is
    an input; this function does not determine its derivative.

    The extended system consists of three coupled equations:

    1. **Classical nodal**: ∂EPI/∂t = νf · ΔNFR(t)
       - Unchanged from original TNFR theory
       - Primary Information Structure evolution

    2. **Phase transport**: ∂θ/∂t = α·νf·sin(π·ΔNFR) + β·ΔNFR + γ·J_φ·κ
       - α: νf-θ coupling (autoorganization)
       - β: ΔNFR sensitivity (pressure response)
       - γ: J_φ transport efficiency
       - κ: coupling_strength (network-dependent)

    3. **Pressure response**: ∂ΔNFR/∂t = -∇·J_ΔNFR - λ·|∇·J_ΔNFR|·sign(∇·J_ΔNFR)
       - This equals -(1+λ) times the supplied divergence.
       - It is not a restoring term proportional to pressure and does not
         ensure relaxation or bounded accumulated pressure.

    The implemented coefficients are α=0.5, β=0.15, γ=0.135 and λ=0.135.
    Compatibility with a graph-derived pressure realization requires its
    separate chain-rule identity; the scalar inputs do not verify it.

    Args:
        nu_f: Structural frequency in Hz_str
        delta_nfr: Nodal gradient (reorganization operator)
        theta: Phase value in radians, normalized to [0, 2π) when validated
        j_phi: Phase current (from compute_phase_current)
        j_dnfr_divergence: Divergence ∇·J_ΔNFR (from compute_dnfr_flux)
        coupling_strength: Nonnegative local transport coefficient; values above one allowed
        validate_units: If True, checks numeric inputs/outputs (not unit calibration)
        graph: Optional graph for context-aware validation

    Returns:
        ExtendedNodalEquationResult with all derivatives and metadata

    Raises:
        FrequencyError: If validated capacity is invalid
        NetworkConfigError: If another validated input or derivative is invalid
        TNFRValueError: If validate_units is not a boolean

    Notes:
        - Zero supplied flux/divergence leaves the original EPI product and
          zero pressure derivative, but the added phase response can remain.
        - Numeric validation certifies neither operator admission nor
          preservation of the canonical invariants or structural tetrad.
        - The optional synchronous-Euler integrator applies this completion;
          ordinary runtime phase coordination is a separate later substep.

    Examples:
        >>> # Zero flux removes transport, but not the chosen phase response.
        >>> result = compute_extended_nodal_system(1.0, 0.5, 0.0, 0.0, 0.0)
        >>> result.classical_derivative  # Should equal 1.0 * 0.5
        0.5
        >>> result.phase_derivative
        0.575
        >>> abs(result.dnfr_derivative)  # Signed zero carries no pressure change.
        0.0

        >>> # With phase transport
        >>> result = compute_extended_nodal_system(1.0, 0.2, 0.5, 0.1, 0.0, 0.8)
        >>> result.j_phi               # Should reflect input
        0.1
        >>> result.coupling_strength   # Should reflect input
        0.8
    """
    nodal = compute_canonical_nodal_derivative(
        nu_f, delta_nfr, validate_units=validate_units, graph=graph
    )
    nu_f, delta_nfr = nodal.nu_f, nodal.delta_nfr

    if validate_units:
        # Validate extended parameters
        theta = _validate_phase(theta)
        j_phi = _validate_flux_field(j_phi, "J_φ")
        j_dnfr_divergence = _validate_flux_divergence(j_dnfr_divergence)
        coupling_strength = _validate_coupling_strength(coupling_strength)

    # 2. Extended phase evolution with J_φ transport
    try:
        phase_derivative = _compute_phase_transport_derivative(
            nu_f, delta_nfr, theta, j_phi, coupling_strength
        )
    except (OverflowError, ValueError) as exc:
        if not validate_units:
            raise
        raise NetworkConfigError(
            parameter="dtheta_dt",
            value=(nu_f, delta_nfr, j_phi, coupling_strength),
            reason="The configured phase response must be representable and finite",
        ) from exc

    # 3. Optional scaled-divergence pressure response
    dnfr_derivative = _compute_dnfr_conservation_derivative(j_dnfr_divergence)
    if validate_units:
        phase_derivative = _validate_real_parameter(phase_derivative, "dtheta_dt")
        dnfr_derivative = _validate_real_parameter(dnfr_derivative, "ddelta_nfr_dt")

    return ExtendedNodalEquationResult(
        classical_derivative=nodal.derivative,
        phase_derivative=phase_derivative,
        dnfr_derivative=dnfr_derivative,
        j_phi=j_phi,
        j_dnfr_divergence=j_dnfr_divergence,
        coupling_strength=coupling_strength,
        validated=nodal.validated,
    )


def _validate_phase(theta: float) -> float:
    """Validate phase parameter for extended dynamics."""
    value = _validate_real_parameter(theta, "phase")

    # Normalize to [0, 2π) range.
    normalized = value % (2 * math.pi)
    return normalized


def _validate_flux_field(flux: float, field_name: str) -> float:
    """Validate flux field (J_φ, J_ΔNFR) for extended dynamics."""
    # Flux fields can be positive (source) or negative (sink)
    return _validate_real_parameter(flux, field_name)


def _validate_flux_divergence(div_j: float) -> float:
    """Validate flux divergence ∇·J for conservation equations."""
    return _validate_real_parameter(div_j, "flux_divergence")


def _validate_coupling_strength(kappa: float) -> float:
    """Validate coupling strength for transport efficiency."""
    value = _validate_real_parameter(kappa, "coupling_strength")

    if value < 0:
        raise NetworkConfigError(
            parameter="coupling_strength",
            value=value,
            reason="Coupling strength must be non-negative",
        )

    # Allow > 1.0 for strong coupling regimes
    return value


def _compute_phase_transport_derivative(
    nu_f: float, delta_nfr: float, theta: float, j_phi: float, coupling_strength: float
) -> float:
    """Evaluate the separately prescribed phase-response law.

    Extended phase equation:
    ∂θ/∂t = α·νf·sin(π·ΔNFR) + β·ΔNFR + γ·J_φ·κ

    Terms:
    - Autoorganization: α·νf·sin(π·ΔNFR) [nonlinear νf-θ coupling]
    - Pressure response: β·ΔNFR [linear response to reorganization]
    - Transport: γ·J_φ·κ [directed flux with coupling efficiency]

    The operational coefficients and this functional form are additional
    premises. The supplied absolute phase is not consumed by this formula.
    """
    # Extended-equation coefficients (optional J_φ-transport path). The term
    # contracts fix each channel and sign; these set the magnitudes: alpha is the
    # unit midpoint; beta/gamma are gentle operational sensitivities on the
    # |ΔNFR| / transport magnitude scale (not coherence levels).
    alpha = 0.5  # unit midpoint (autoorganization νf-θ coupling)
    beta = 0.15  # gentle pressure-response sensitivity (operational)
    gamma = 0.135  # gentle transport efficiency (operational)

    # Autoorganization term: nonlinear νf-θ coupling
    autoorg_term = alpha * nu_f * math.sin(math.pi * delta_nfr)

    # Pressure response: linear ΔNFR sensitivity
    pressure_term = beta * delta_nfr

    # Transport term: directed J_φ flux
    transport_term = gamma * j_phi * coupling_strength

    return autoorg_term + pressure_term + transport_term


def _compute_dnfr_conservation_derivative(j_dnfr_divergence: float) -> float:
    """Evaluate the optional scaled-divergence pressure response.

    Conservation equation:
    ∂ΔNFR/∂t = -∇·J_ΔNFR - λ·|∇·J_ΔNFR|·sign(∇·J_ΔNFR)

    For real divergence d, |d| sign(d)=d, so the implemented rate is
    -(1+λ)d. Constant nonzero divergence produces a constant nonzero pressure
    slope; no pressure-dependent restoring term or decay theorem is supplied.
    A realized graph flow needs a separate consistency and balance check.
    """
    # Operational multiplier on supplied divergence, not pressure damping.
    decay_rate = 0.135

    # Base signed divergence response; graph-level balance is a separate claim.
    conservation_term = -j_dnfr_divergence

    # Historical variable name: this rescales divergence without restoring pressure.
    decay_term = (
        -decay_rate * abs(j_dnfr_divergence) * math.copysign(1.0, j_dnfr_divergence)
    )

    return conservation_term + decay_term


# ============================================================================
# UNIFIED NODAL EQUATION INTEGRATION (CANONICAL ENTRY POINT)
# ============================================================================


def integrate_canonical_nodal_equation(
    G: Any,
    *,
    dt: float | None = None,
    method: str = "rk4",
    max_steps: int | None = None,
    tolerance: float | None = None,
    use_gpu: bool | None = None,
) -> dict[str, Any]:
    """Integrate the stored constant-frequency, constant-pressure nodal field.

    This convenience API holds nu_f and DeltaNFR fixed throughout its loop.
    Both accepted method names therefore use the same represented Euler map
    for this constant derivative. It does not recompute pressure from changing
    EPI or supersede the runtime integrator's forcing and boundary policies.

    Parameters
    ----------
    G : TNFRGraph
        Graph with TNFR node attributes (EPI, νf, ΔNFR, phase)
    dt : float, optional
        Finite nonnegative timestep (from config if None); zero is a no-op
    method : {"euler", "rk4"}, default="rk4"
        Integration method
    max_steps : int, optional
        Maximum integration steps (from config if None)
    tolerance : float, optional
        Nonnegative step-change tolerance (from config if None); zero disables
        early stopping. This is not a structural-equilibrium residual.
    use_gpu : bool, optional
        Enable GPU acceleration (from config if None)

    Returns
    -------
    dict[str, Any]
        Integration results with metadata

    Notes
    -----
    ``final_error`` is the norm of the most recent EPI increment. ``converged``
    reports that step-change criterion, not DeltaNFR equilibrium. No convergence
    assessment is made when dt is zero (steps=0, converged=False).
    The compatibility fallback for absent capacity is 1.0; absent form and
    pressure default to zero. These are supplied initialization policies.
    Invalid authoritative inputs and nonfinite candidate output reject before
    any EPI is committed. This API writes neither a clock nor derivative history.
    Numeric controls use the shared finite-real representation contract; returned
    parameters record those materialized values rather than their input objects.
    """
    from ..backend_config import get_config

    # Get configuration defaults (backend_config provides the @dataclass TNFRConfig)
    config = get_config()
    integration_config = config.get_integration_config()

    # Resolve parameters from config
    dt = integration_config["dt"] if dt is None else dt
    max_steps = integration_config["max_steps"] if max_steps is None else max_steps
    tolerance = integration_config["tolerance"] if tolerance is None else tolerance
    use_gpu = use_gpu if use_gpu is not None else (config.gpu_mode != "disabled")

    # Validate inputs
    def nonnegative_parameter(value: Any, label: str) -> float:
        try:
            resolved, _ = finite_represented_real(value, label)
        except (TypeError, ValueError) as exc:
            raise TNFRValueError(str(exc), context={label: value}) from exc
        if resolved < 0:
            raise TNFRValueError(f"{label} must be nonnegative", context={label: value})
        return resolved

    dt_resolved = nonnegative_parameter(dt, "dt")
    tolerance_resolved = nonnegative_parameter(tolerance, "tolerance")
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, Integral)
        or max_steps <= 0
    ):
        raise TNFRValueError(
            f"Max steps must be a positive integer, got {max_steps}",
            context={"max_steps": max_steps},
            suggestion="set max_steps to a positive integer.",
        )

    if not isinstance(use_gpu, bool):
        raise TNFRValueError("use_gpu must be a boolean", context={"use_gpu": use_gpu})
    if method not in ("euler", "rk4"):
        raise TNFRValueError(
            "Integration method must be 'euler' or 'rk4'", context={"method": method}
        )
    max_steps_resolved = int(max_steps)

    # Define GPU and CPU integration functions
    def gpu_integration() -> dict[str, Any]:
        """GPU-accelerated integration using unified backend."""
        from ..mathematics.backend import get_backend

        backend = get_backend()

        # Use backend for accelerated computation
        return _integrate_with_backend(
            G, dt_resolved, method, max_steps_resolved, tolerance_resolved, backend
        )

    def cpu_integration() -> dict[str, Any]:
        """CPU fallback integration using NumPy."""
        from ..mathematics.backend import get_backend

        backend = get_backend("numpy")

        return _integrate_with_backend(
            G, dt_resolved, method, max_steps_resolved, tolerance_resolved, backend
        )

    # Execute with automatic GPU fallback
    if dt_resolved == 0.0:
        result = {"converged": False, "steps": 0, "final_error": 0.0, "time_ms": 0.0}
        backend_used = "none"
    elif use_gpu:
        from ..engines.computation.unified_gpu_system import execute_with_gpu_fallback

        result, backend_used = execute_with_gpu_fallback(
            gpu_integration, cpu_integration
        )
    else:
        result, backend_used = cpu_integration(), "cpu"

    # Add metadata
    result["backend_used"] = backend_used
    result["parameters"] = {
        "dt": dt_resolved,
        "method": method,
        "max_steps": max_steps_resolved,
        "tolerance": tolerance_resolved,
    }

    return result


def _integrate_with_backend(
    G: Any, dt: float, method: str, max_steps: int, tolerance: float, backend: Any
) -> dict[str, Any]:
    """Internal integration implementation using specified backend."""
    import time

    from ._euler_kernel import euler_update

    start_time = time.perf_counter()

    # Extract node states
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    if n_nodes == 0:
        return {"converged": True, "steps": 0, "final_error": 0.0, "time_ms": 0.0}

    # Initialize arrays using backend (canonical alias-aware reads:
    # honour Greek primaries 'νf'/'ΔNFR' as well as ASCII aliases)
    epi_values = backend.as_array(
        [
            get_attr(
                G.nodes[node],
                ALIAS_EPI,
                0.0,
                strict=True,
                conv=require_finite_real_scalar_epi,
            )
            for node in nodes
        ]
    )
    # Validate authoritative aliases before device arithmetic can hide invalid
    # capacity, pressure or an overflowing nodal product.
    derivative_values = [
        compute_canonical_nodal_derivative(
            get_attr(
                G.nodes[node],
                ALIAS_VF,
                1.0,
                strict=True,
                conv=validate_structural_frequency,
            ),
            get_attr(
                G.nodes[node],
                ALIAS_DNFR,
                0.0,
                strict=True,
                conv=validate_nodal_gradient,
            ),
        ).derivative
        for node in nodes
    ]
    # The stored fields are frozen, so the legacy RK4 name uses this same slope.
    derivatives = backend.as_array(derivative_values)
    converged = False
    final_error = float("inf")

    for step in range(max_steps):
        # Store previous values
        epi_prev = epi_values

        epi_values = euler_update(epi_prev, dt, derivatives)
        # Stage and check every node before the final graph-owned commit.
        epi_final = tuple(
            require_finite_real_scalar_epi(value)
            for value in backend.to_numpy(epi_values)
        )
        # hypot avoids overflow from squaring otherwise representable increments.
        diff = backend.to_numpy(epi_values - epi_prev)
        final_error = math.hypot(*(float(value) for value in diff))
        if not math.isfinite(final_error):
            raise NetworkConfigError(
                parameter="final_error",
                value=final_error,
                reason="The represented step-change norm must be finite",
            )

        if final_error < tolerance:
            converged = True
            break

    # Update graph with final values (canonical alias-aware write)
    for i, node in enumerate(nodes):
        set_attr(G.nodes[node], ALIAS_EPI, float(epi_final[i]))

    end_time = time.perf_counter()

    return {
        "converged": converged,
        "steps": step + 1,
        "final_error": final_error,
        "time_ms": (end_time - start_time) * 1000.0,
    }
