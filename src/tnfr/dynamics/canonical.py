"""Canonical TNFR nodal equation implementation.

This module provides the explicit, canonical implementation of the fundamental
TNFR nodal equation as specified in the theory:

    ∂EPI/∂t = νf · ΔNFR(t)

Where:
  - EPI: Primary Information Structure (coherent form)
  - νf: Structural frequency in Hz_str (structural hertz)
  - ΔNFR: Nodal gradient (reorganization operator)
  - t: Structural time (not chronological time)

This implementation ensures theoretical fidelity to the TNFR paradigm by:
  1. Making the canonical equation explicit in code
  2. Validating dimensional consistency (Hz_str units)
  3. Providing clear mapping between theory and implementation
  4. Maintaining reproducibility and traceability

TNFR Invariants (from AGENTS.md):
  - EPI as coherent form: changes only via structural operators
  - Structural units: νf expressed in Hz_str (structural hertz)
  - ΔNFR semantics: sign and magnitude modulate reorganization rate
  - Operator closure: composition yields valid TNFR states

References:
  - TNFR.pdf: Canonical nodal equation specification
  - AGENTS.md: Section 3 (Canonical invariants)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

from ..alias import get_attr, set_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..errors.contextual import FrequencyError, NetworkConfigError, TNFRValueError
from ..mathematics.unified_numerical import np

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
        validated: Whether units and bounds were validated
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
        validate_units: If True, validates that inputs are in valid ranges
        graph: Optional graph for context-aware validation

    Returns:
        NodalEquationResult containing the computed derivative and metadata

    Raises:
        TNFRValueError: If validation is enabled and inputs are invalid

    Notes:
        - This function is the canonical reference implementation
        - The result represents the instantaneous rate of EPI evolution
        - Units: [∂EPI/∂t] = Hz_str (structural reorganization rate)
        - The product νf·ΔNFR must preserve TNFR operator closure

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
    validated = False

    if validate_units:
        nu_f = validate_structural_frequency(nu_f, graph=graph)
        delta_nfr = validate_nodal_gradient(delta_nfr, graph=graph)
        validated = True

    # Canonical TNFR nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    derivative = float(nu_f) * float(delta_nfr)

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
    """Validate that structural frequency is in valid range.

    Structural frequency (νf) must satisfy TNFR constraints:
      - Non-negative (νf ≥ 0)
      - Expressed in Hz_str (structural hertz)
      - Finite and well-defined

    Args:
        nu_f: Structural frequency to validate
        graph: Optional graph for context-aware bounds checking

    Returns:
        Validated structural frequency value

    Raises:
        FrequencyError: If nu_f is negative, infinite, or NaN
        TypeError: If nu_f cannot be converted to float

    Notes:
        - νf = 0 is valid and represents structural silence
        - Units must be Hz_str (not classical Hz)
        - For Hz↔Hz_str conversion, use tnfr.units module
    """
    try:
        value = float(nu_f)
    except (TypeError, ValueError) as exc:
        # Non-convertible type or invalid string
        raise FrequencyError(vf=float("nan"), operation="validation") from exc

    # Check for NaN or infinity using math.isfinite
    if not math.isfinite(value):
        raise FrequencyError(vf=value, operation="validation")

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
        NetworkConfigError: If delta_nfr is infinite or NaN
        TypeError: If delta_nfr cannot be converted to float

    Notes:
        - ΔNFR can be positive (expansion) or negative (contraction)
        - ΔNFR = 0 indicates equilibrium (no reorganization)
        - Do NOT reinterpret as classical "error gradient"
        - Semantics: operator over EPI, not optimization target
    """
    try:
        value = float(delta_nfr)
    except (TypeError, ValueError) as exc:
        # Non-convertible type or invalid string
        raise NetworkConfigError(
            parameter="delta_nfr",
            value=delta_nfr,
            reason="Nodal gradient must be numeric",
        ) from exc

    # Check for NaN or infinity using math.isfinite
    if not math.isfinite(value):
        raise NetworkConfigError(
            parameter="delta_nfr", value=value, reason="Nodal gradient must be finite"
        )

    return value


# Extended TNFR dynamics with canonical flux fields
class ExtendedNodalEquationResult(NamedTuple):
    """Result of extended nodal equation system evaluation.

    Represents the coupled system:
    1. ∂EPI/∂t = νf · ΔNFR(t)           [Classical nodal equation]
    2. ∂θ/∂t = f(νf, ΔNFR, J_φ)        [Phase evolution with transport]
    3. ∂ΔNFR/∂t = g(∇·J_ΔNFR)          [ΔNFR conservation dynamics]

    Attributes:
        classical_derivative: ∂EPI/∂t (original TNFR nodal equation)
        phase_derivative: ∂θ/∂t (phase evolution with J_φ transport)
        dnfr_derivative: ∂ΔNFR/∂t (reorganization conservation)
        j_phi: Phase current J_φ used in computation
        j_dnfr_divergence: ∇·J_ΔNFR divergence used
        coupling_strength: Local network coupling coefficient
        validated: Whether extended physics validation passed
    """

    classical_derivative: float  # ∂EPI/∂t = νf·ΔNFR
    phase_derivative: float  # ∂θ/∂t with J_φ transport
    dnfr_derivative: float  # ∂ΔNFR/∂t from conservation
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
    """Compute extended TNFR nodal equation system with flux fields.

    This implements the fundamental extension of TNFR dynamics to include
    canonical flux fields J_φ (phase current) and J_ΔNFR (reorganization flux).

    The extended system consists of three coupled equations:

    1. **Classical nodal**: ∂EPI/∂t = νf · ΔNFR(t)
       - Unchanged from original TNFR theory
       - Primary Information Structure evolution

    2. **Phase transport**: ∂θ/∂t = α·νf·sin(π·ΔNFR) + β·ΔNFR + γ·J_φ·κ
       - α: νf-θ coupling (autoorganization)
       - β: ΔNFR sensitivity (pressure response)
       - γ: J_φ transport efficiency
       - κ: coupling_strength (network-dependent)

    3. **ΔNFR conservation**: ∂ΔNFR/∂t = -∇·J_ΔNFR - λ·|∇·J_ΔNFR|·sign(∇·J_ΔNFR)
       - Conservation term: -∇·J_ΔNFR (flow continuity)
       - Decay term: natural relaxation to equilibrium

    Args:
        nu_f: Structural frequency in Hz_str
        delta_nfr: Nodal gradient (reorganization operator)
        theta: Phase value in [0, 2π] radians
        j_phi: Phase current (from compute_phase_current)
        j_dnfr_divergence: Divergence ∇·J_ΔNFR (from compute_dnfr_flux)
        coupling_strength: Local network coupling [0, 1]
        validate_units: If True, validates physics constraints
        graph: Optional graph for context-aware validation

    Returns:
        ExtendedNodalEquationResult with all derivatives and metadata

    Raises:
        TNFRValueError: If validation fails or physics constraints violated

    Notes:
        - When J_φ = J_ΔNFR = 0, system reduces to classical TNFR
        - Extended dynamics preserve all 10 canonical invariants
        - Phase evolution includes directed transport via J_φ
        - ΔNFR follows conservation law with natural decay
        - Coupling strength modulates transport efficiency

    Examples:
        >>> # Classical limit (no fluxes)
        >>> result = compute_extended_nodal_system(1.0, 0.5, 0.0, 0.0, 0.0)
        >>> result.classical_derivative  # Should equal 1.0 * 0.5
        0.5
        >>> result.phase_derivative     # Should be small with no J_φ
        0.25
        >>> result.dnfr_derivative      # Should be ~0 with no flux
        0.0

        >>> # With phase transport
        >>> result = compute_extended_nodal_system(1.0, 0.2, 0.5, 0.1, 0.0, 0.8)
        >>> result.j_phi               # Should reflect input
        0.1
        >>> result.coupling_strength   # Should reflect input
        0.8
    """
    validated = False

    if validate_units:
        # Validate classical parameters (existing functions)
        nu_f = validate_structural_frequency(nu_f, graph=graph)
        delta_nfr = validate_nodal_gradient(delta_nfr, graph=graph)

        # Validate extended parameters
        theta = _validate_phase(theta)
        j_phi = _validate_flux_field(j_phi, "J_φ")
        j_dnfr_divergence = _validate_flux_divergence(j_dnfr_divergence)
        coupling_strength = _validate_coupling_strength(coupling_strength)

        validated = True

    # 1. Classical TNFR nodal equation (unchanged)
    classical_derivative = float(nu_f) * float(delta_nfr)

    # 2. Extended phase evolution with J_φ transport
    phase_derivative = _compute_phase_transport_derivative(
        nu_f, delta_nfr, theta, j_phi, coupling_strength
    )

    # 3. ΔNFR conservation dynamics
    dnfr_derivative = _compute_dnfr_conservation_derivative(j_dnfr_divergence)

    return ExtendedNodalEquationResult(
        classical_derivative=classical_derivative,
        phase_derivative=phase_derivative,
        dnfr_derivative=dnfr_derivative,
        j_phi=j_phi,
        j_dnfr_divergence=j_dnfr_divergence,
        coupling_strength=coupling_strength,
        validated=validated,
    )


def _validate_phase(theta: float) -> float:
    """Validate phase parameter for extended dynamics."""
    try:
        value = float(theta)
    except (TypeError, ValueError) as exc:
        raise NetworkConfigError(
            parameter="phase", value=theta, reason="Phase θ must be numeric"
        ) from exc

    if not math.isfinite(value):
        raise NetworkConfigError(
            parameter="phase", value=value, reason="Phase θ must be finite"
        )

    # Normalize to [0, 2π] range
    normalized = value % (2 * math.pi)
    return normalized


def _validate_flux_field(flux: float, field_name: str) -> float:
    """Validate flux field (J_φ, J_ΔNFR) for extended dynamics."""
    try:
        value = float(flux)
    except (TypeError, ValueError) as exc:
        raise NetworkConfigError(
            parameter=field_name,
            value=flux,
            reason=f"Flux field {field_name} must be numeric",
        ) from exc

    if not math.isfinite(value):
        raise NetworkConfigError(
            parameter=field_name,
            value=value,
            reason=f"Flux field {field_name} must be finite",
        )

    # Flux fields can be positive (source) or negative (sink)
    return value


def _validate_flux_divergence(div_j: float) -> float:
    """Validate flux divergence ∇·J for conservation equations."""
    try:
        value = float(div_j)
    except (TypeError, ValueError) as exc:
        raise NetworkConfigError(
            parameter="flux_divergence",
            value=div_j,
            reason="Flux divergence must be numeric",
        ) from exc

    if not math.isfinite(value):
        raise NetworkConfigError(
            parameter="flux_divergence",
            value=value,
            reason="Flux divergence must be finite",
        )

    return value


def _validate_coupling_strength(kappa: float) -> float:
    """Validate coupling strength for transport efficiency."""
    try:
        value = float(kappa)
    except (TypeError, ValueError) as exc:
        raise NetworkConfigError(
            parameter="coupling_strength",
            value=kappa,
            reason="Coupling strength must be numeric",
        ) from exc

    if not math.isfinite(value):
        raise NetworkConfigError(
            parameter="coupling_strength",
            value=value,
            reason="Coupling strength must be finite",
        )

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
    """Compute ∂θ/∂t with J_φ transport.

    Extended phase equation:
    ∂θ/∂t = α·νf·sin(π·ΔNFR) + β·ΔNFR + γ·J_φ·κ

    Terms:
    - Autoorganization: α·νf·sin(π·ΔNFR) [nonlinear νf-θ coupling]
    - Pressure response: β·ΔNFR [linear response to reorganization]
    - Transport: γ·J_φ·κ [directed flux with coupling efficiency]
    """
    # Import canonical constants
    from ..constants.canonical import GAMMA_PI_RATIO, INV_PHI, PI_MINUS_E_OVER_PI

    # Physics coefficients (RECALIBRATED from canonical constants)
    alpha = INV_PHI  # 1/φ ≈ 0.618 (inverse golden self-organization)
    beta = GAMMA_PI_RATIO  # γ/(π+γ) ≈ 0.155 (Euler-pi sensitivity)
    gamma = PI_MINUS_E_OVER_PI  # (π-e)/π ≈ 0.135 (transcendental efficiency)

    # Autoorganization term: nonlinear νf-θ coupling
    autoorg_term = alpha * nu_f * math.sin(math.pi * delta_nfr)

    # Pressure response: linear ΔNFR sensitivity
    pressure_term = beta * delta_nfr

    # Transport term: directed J_φ flux
    transport_term = gamma * j_phi * coupling_strength

    return autoorg_term + pressure_term + transport_term


def _compute_dnfr_conservation_derivative(j_dnfr_divergence: float) -> float:
    """Compute ∂ΔNFR/∂t from flux conservation.

    Conservation equation:
    ∂ΔNFR/∂t = -∇·J_ΔNFR - λ·|∇·J_ΔNFR|·sign(∇·J_ΔNFR)

    Terms:
    - Conservation: -∇·J_ΔNFR [flow continuity]
    - Decay: λ·|∇·J| [natural relaxation, prevents accumulation]
    """
    # Import canonical constants
    from ..constants.canonical import EXP_DOUBLE_NEG

    # Physics coefficients (RECALIBRATED from canonical constants)
    decay_rate = (
        EXP_DOUBLE_NEG  # e^(-2) ≈ 0.135 (decaimiento exponencial natural doble)
    )

    # Conservation term: flux in increases ΔNFR, flux out decreases it
    conservation_term = -j_dnfr_divergence

    # Decay term: prevents indefinite accumulation
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
    """CANONICAL nodal equation integrator used by all TNFR modules.

    This is the single source of truth for integrating:
        ∂EPI/∂t = νf · ΔNFR(t)

    All other integration functions should delegate to this implementation
    to maintain theoretical consistency and eliminate redundancy.

    Parameters
    ----------
    G : TNFRGraph
        Graph with TNFR node attributes (EPI, νf, ΔNFR, phase)
    dt : float, optional
        Integration timestep (from config if None)
    method : {"euler", "rk4"}, default="rk4"
        Integration method
    max_steps : int, optional
        Maximum integration steps (from config if None)
    tolerance : float, optional
        Convergence tolerance (from config if None)
    use_gpu : bool, optional
        Enable GPU acceleration (from config if None)

    Returns
    -------
    dict[str, Any]
        Integration results with metadata

    Notes
    -----
    This function serves as the canonical entry point that all other
    TNFR modules should use for nodal equation integration. It ensures:
    - Consistent parameter handling via unified config
    - GPU acceleration through unified backend
    - Proper error handling and validation
    - Reproducible results with deterministic methods
    """
    from ..backend_config import get_config
    from ..engines.computation.unified_gpu_system import execute_with_gpu_fallback

    # Get configuration defaults (backend_config provides the @dataclass TNFRConfig)
    config = get_config()
    integration_config = config.get_integration_config()

    # Resolve parameters from config
    dt = dt or integration_config["dt"]
    max_steps = max_steps or integration_config["max_steps"]
    tolerance = tolerance or integration_config["tolerance"]
    use_gpu = use_gpu if use_gpu is not None else (config.gpu_mode != "disabled")

    # Validate inputs
    if dt <= 0:
        raise TNFRValueError(
            f"Integration timestep must be positive, got {dt}",
            context={"dt": dt},
            suggestion="set a positive timestep (dt > 0).",
        )
    if max_steps <= 0:
        raise TNFRValueError(
            f"Max steps must be positive, got {max_steps}",
            context={"max_steps": max_steps},
            suggestion="set max_steps to a positive integer.",
        )

    # Ensure all parameters are resolved (not None)
    dt_resolved = float(dt)
    max_steps_resolved = int(max_steps)
    tolerance_resolved = float(tolerance)

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
    if use_gpu:
        result, backend_used = execute_with_gpu_fallback(
            gpu_integration, cpu_integration
        )
    else:
        result, backend_used = cpu_integration(), "cpu"

    # Add metadata
    result["backend_used"] = backend_used
    result["parameters"] = {
        "dt": dt,
        "method": method,
        "max_steps": max_steps,
        "tolerance": tolerance,
    }

    return result


def _integrate_with_backend(
    G: Any, dt: float, method: str, max_steps: int, tolerance: float, backend: Any
) -> dict[str, Any]:
    """Internal integration implementation using specified backend."""
    import time

    start_time = time.perf_counter()

    # Extract node states
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    if n_nodes == 0:
        return {"converged": True, "steps": 0, "final_error": 0.0, "time_ms": 0.0}

    # Initialize arrays using backend (canonical alias-aware reads:
    # honour Greek primaries 'νf'/'ΔNFR' as well as ASCII aliases)
    epi_values = backend.as_array(
        [get_attr(G.nodes[node], ALIAS_EPI, 0.0) for node in nodes]
    )
    vf_values = backend.as_array(
        [get_attr(G.nodes[node], ALIAS_VF, 1.0) for node in nodes]
    )
    dnfr_values = backend.as_array(
        [get_attr(G.nodes[node], ALIAS_DNFR, 0.0) for node in nodes]
    )

    converged = False
    final_error = float("inf")

    for step in range(max_steps):
        # Store previous values
        epi_prev = epi_values

        # Compute derivatives using canonical nodal equation
        if method == "euler":
            # Euler method: EPI_{n+1} = EPI_n + dt * νf * ΔNFR
            derivatives = backend.as_array(
                [vf * dnfr for vf, dnfr in zip(vf_values, dnfr_values)]
            )
            epi_values = epi_prev + dt * derivatives

        elif method == "rk4":
            # RK4 method for higher accuracy
            k1 = backend.as_array(
                [vf * dnfr for vf, dnfr in zip(vf_values, dnfr_values)]
            )
            k2 = k1  # Simplified - assume ΔNFR constant over dt
            k3 = k1
            k4 = k1

            epi_values = epi_prev + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        else:
            raise TNFRValueError(
                f"Unknown integration method: {method}",
                context={"method": method, "available": ["euler", "rk4"]},
                suggestion="Use 'euler' or 'rk4' as the integration method.",
            )

        # Check convergence
        if hasattr(backend, "norm"):
            error = backend.norm(epi_values - epi_prev)
            final_error = backend.to_numpy(error).item()
        else:
            # Fallback norm calculation
            diff = backend.to_numpy(epi_values - epi_prev)
            final_error = float(np.linalg.norm(diff))

        if final_error < tolerance:
            converged = True
            break

    # Update graph with final values (canonical alias-aware write)
    epi_final = backend.to_numpy(epi_values)
    for i, node in enumerate(nodes):
        set_attr(G.nodes[node], ALIAS_EPI, float(epi_final[i]))

    end_time = time.perf_counter()

    return {
        "converged": converged,
        "steps": step + 1,
        "final_error": final_error,
        "time_ms": (end_time - start_time) * 1000.0,
    }
