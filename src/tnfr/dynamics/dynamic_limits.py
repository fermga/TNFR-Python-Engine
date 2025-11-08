"""Dynamic canonical limits based on network coherence.

This module implements dynamic canonical limits for EPI and νf that adapt
based on network coherence metrics. This addresses the theoretical question
of whether fixed limits contradict TNFR's self-organizing principles.

Theoretical Foundation
----------------------
TNFR paradigm principles:
1. **Operational fractality**: Patterns should scale without artificial bounds
2. **Self-organization**: System should find its own natural limits
3. **Coherence emergence**: Stability arises from resonance, not external constraints

Fixed limits may contradict these principles by imposing external constraints
that don't emerge from the system's own dynamics. Dynamic limits address this
by making bounds proportional to network coherence metrics.

Dynamic Limit Formulas
----------------------
EPI effective maximum:
    EPI_effective_max(t) = EPI_base_max × (1 + α × C(t) × Si_avg)

νf effective maximum:
    νf_effective_max(t) = νf_base_max × (1 + β × R_kuramoto)

Where:
- C(t): Global coherence at time t (0 to 1)
- Si_avg: Average sense index across network (0 to 1+)
- R_kuramoto: Kuramoto order parameter (0 to 1)
- α, β: Expansion coefficients (default: 0.5, 0.3)

Theoretical Justification
--------------------------
1. **Coherent networks** (high C(t), Si) can sustain higher EPI values
2. **Synchronized networks** (high R_kuramoto) can sustain higher νf
3. **Self-regulation** through coupling to system state
4. **Fractality preservation** via proportional scaling
5. **Safety bounds** via maximum expansion factors

References
----------
- AGENTS.md: Section 3 (Canonical invariants)
- Issue fermga/TNFR-Python-Engine#2624: Theoretical review of limits
- TNFR.pdf: Nodal equation and coherence theory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..alias import collect_attr
from ..constants.aliases import ALIAS_SI
from ..metrics.common import compute_coherence
from ..observers import kuramoto_order
from ..utils import get_numpy

if TYPE_CHECKING:
    from ..types import TNFRGraph

__all__ = (
    "DynamicLimits",
    "DynamicLimitsConfig",
    "compute_dynamic_limits",
    "DEFAULT_SI_FALLBACK",
)

# Default fallback value for sense index when nodes have no Si attribute
# This represents a "neutral" sense index - neither high stability nor instability
DEFAULT_SI_FALLBACK = 0.5


@dataclass(frozen=True)
class DynamicLimitsConfig:
    """Configuration for dynamic canonical limits.

    Attributes:
        base_epi_max: Base maximum for EPI (static fallback)
        base_vf_max: Base maximum for νf (static fallback)
        alpha: Expansion coefficient for EPI (via C(t) × Si)
        beta: Expansion coefficient for νf (via R_kuramoto)
        max_expansion_factor: Maximum allowed expansion multiplier
        enabled: Whether dynamic limits are active
    """

    base_epi_max: float = 1.0
    base_vf_max: float = 10.0
    alpha: float = 0.5
    beta: float = 0.3
    max_expansion_factor: float = 3.0
    enabled: bool = True


@dataclass(frozen=True)
class DynamicLimits:
    """Result of dynamic limits computation.

    Attributes:
        epi_max_effective: Computed effective maximum for EPI
        vf_max_effective: Computed effective maximum for νf
        coherence: C(t) value used in computation
        si_avg: Average Si value used in computation
        kuramoto_r: Kuramoto order parameter used in computation
        coherence_factor: Combined coherence metric (C(t) × Si_avg)
        config: Configuration used for computation
    """

    epi_max_effective: float
    vf_max_effective: float
    coherence: float
    si_avg: float
    kuramoto_r: float
    coherence_factor: float
    config: DynamicLimitsConfig


def compute_dynamic_limits(
    G: TNFRGraph,
    config: DynamicLimitsConfig | None = None,
) -> DynamicLimits:
    """Compute dynamic canonical limits based on network state.

    This function computes effective maximum values for EPI and νf that
    adapt based on the network's coherence metrics. More coherent networks
    are allowed higher values, reflecting their greater structural stability.

    The computation respects TNFR invariants:
    - Operator closure: bounds scale but remain finite
    - Structural semantics: expansion proportional to coherence
    - Self-organization: limits emerge from system state
    - Fractality: proportional scaling preserves structure

    Args:
        G: TNFR graph with node attributes
        config: Configuration for dynamic limits (uses defaults if None)

    Returns:
        DynamicLimits containing effective bounds and metrics

    Examples:
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> G.add_node(0, **{"νf": 1.0, "theta": 0.0, "EPI": 0.5, "Si": 0.7})
        >>> limits = compute_dynamic_limits(G)
        >>> limits.epi_max_effective >= 1.0  # May be expanded
        True

        >>> # With custom configuration
        >>> config = DynamicLimitsConfig(alpha=0.8, beta=0.5)
        >>> limits = compute_dynamic_limits(G, config)
        >>> limits.config.alpha
        0.8

    Notes:
        - Returns base limits if graph is empty
        - Clamps expansion to max_expansion_factor for safety
        - If config.enabled is False, returns base limits only
        - All metrics computed using existing TNFR functions
    """
    if config is None:
        config = DynamicLimitsConfig()

    # Handle empty graph
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return DynamicLimits(
            epi_max_effective=config.base_epi_max,
            vf_max_effective=config.base_vf_max,
            coherence=0.0,
            si_avg=0.0,
            kuramoto_r=0.0,
            coherence_factor=0.0,
            config=config,
        )

    # If dynamic limits disabled, return base values
    if not config.enabled:
        return DynamicLimits(
            epi_max_effective=config.base_epi_max,
            vf_max_effective=config.base_vf_max,
            coherence=1.0,
            si_avg=1.0,
            kuramoto_r=1.0,
            coherence_factor=1.0,
            config=config,
        )

    # Compute coherence metrics
    C_t = compute_coherence(G)

    # Compute average sense index
    np_module = get_numpy()
    si_values = collect_attr(G, G.nodes, ALIAS_SI, DEFAULT_SI_FALLBACK, np=np_module)
    if np_module is not None:
        Si_avg = float(np_module.mean(si_values))
    else:
        Si_avg = sum(si_values) / len(si_values) if si_values else DEFAULT_SI_FALLBACK

    # Compute Kuramoto order parameter
    R_kuramoto = kuramoto_order(G)

    # Compute coherence factor
    coherence_factor = C_t * Si_avg

    # Compute dynamic limits
    epi_expansion = 1.0 + config.alpha * coherence_factor
    vf_expansion = 1.0 + config.beta * R_kuramoto

    # Apply maximum expansion factor as safety bound
    epi_expansion = min(epi_expansion, config.max_expansion_factor)
    vf_expansion = min(vf_expansion, config.max_expansion_factor)

    # Compute effective limits
    epi_max_effective = config.base_epi_max * epi_expansion
    vf_max_effective = config.base_vf_max * vf_expansion

    return DynamicLimits(
        epi_max_effective=epi_max_effective,
        vf_max_effective=vf_max_effective,
        coherence=C_t,
        si_avg=Si_avg,
        kuramoto_r=R_kuramoto,
        coherence_factor=coherence_factor,
        config=config,
    )
