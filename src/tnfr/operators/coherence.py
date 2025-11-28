"""Coherence (IL) operator.

Purpose: stabilize form; reduce delta NFR; raise coherence.
Physics: negative feedback drives delta NFR toward 0.
Grammar: stabilizer (U2); safe closure component.
Effects: pressure lowers; EPI preserved; optional phase locking.
Preconditions: active structure; prior destabilizer if recent IL already.
Typical: AL->UM->IL; OZ->IL; VAL->IL. Avoid redundant IL chains.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import COHERENCE
from ..constants.aliases import ALIAS_DNFR
from ..types import Glyph, TNFRGraph
from ..utils import get_numpy
from .definitions_base import Operator


class Coherence(Operator):
    """Stabilize alignment; compress delta NFR; boost coherence.

    Drives local equilibrium (delta NFR down). Often follows
    Emission/Coupling or contains OZ/VAL effects. Phase locking optional.
    """

    __slots__ = ()
    name: ClassVar[str] = COHERENCE
    glyph: ClassVar[Glyph] = Glyph.IL

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Reduce delta NFR; track coherence; optional phase lock.

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes and structural operator history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Optional keys:
            - coherence_radius: local coherence radius (default 1)
            - phase_locking_coefficient: phase lock strength in [0.1,0.5]

        Notes
        -----
        Canonical effect: dnfr -> dnfr * (1 - rho) rho≈0.3.

        Reduction applied by grammar layer (dnfr_factor). Adds telemetry.

        **C(t) Coherence Tracking:**

        Captures global/local coherence before & after:
        - C_global: 1 - (sigma_dnfr / dnfr_max)
        - C_local: neighborhood coherence (radius)

        Stored in G.graph['IL_coherence_tracking'].

        Phase Locking:

        Phase align:
        - theta_new = theta + a*(theta_net - theta)
        Stored in telemetry list.

        Adjust rho via GLYPH_FACTORS['IL_dnfr_factor'] (default 0.7).
        """
        # Import here to avoid circular import
        from ..metrics.coherence import (
            compute_global_coherence,
            compute_local_coherence,
        )

        # Capture C(t) before Coherence application
        C_global_before = compute_global_coherence(G)
        C_local_before = compute_local_coherence(
            G,
            node,
            radius=kw.get("coherence_radius", 1),
        )

        # Capture ΔNFR before Coherence application for telemetry
        dnfr_before = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

        # Parent __call__ applies grammar (includes reduction)
        super().__call__(G, node, **kw)

        # Apply phase locking after grammar application
        locking_coef = kw.get("phase_locking_coefficient", 0.3)
        self._apply_phase_locking(G, node, locking_coefficient=locking_coef)

        # Capture C(t) after IL application
        C_global_after = compute_global_coherence(G)
        C_local_after = compute_local_coherence(
            G,
            node,
            radius=kw.get("coherence_radius", 1),
        )

        # Capture ΔNFR after IL application for telemetry
        dnfr_after = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

        # Store C(t) tracking in graph telemetry
        if "IL_coherence_tracking" not in G.graph:
            G.graph["IL_coherence_tracking"] = []

        G.graph["IL_coherence_tracking"].append(
            {
                "node": node,
                "C_global_before": C_global_before,
                "C_global_after": C_global_after,
                "C_global_delta": C_global_after - C_global_before,
                "C_local_before": C_local_before,
                "C_local_after": C_local_after,
                "C_local_delta": C_local_after - C_local_before,
            }
        )

        # Log ΔNFR reduction in graph metadata for telemetry
        if "IL_dnfr_reductions" not in G.graph:
            G.graph["IL_dnfr_reductions"] = []

        # Calculate actual reduction factor from before/after values
        if dnfr_before > 0:
            actual_reduction_factor = (dnfr_before - dnfr_after) / dnfr_before
        else:
            actual_reduction_factor = 0.0

        G.graph["IL_dnfr_reductions"].append(
            {
                "node": node,
                "before": dnfr_before,
                "after": dnfr_after,
                "reduction": dnfr_before - dnfr_after,
                "reduction_factor": actual_reduction_factor,
            }
        )

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate IL-specific preconditions."""
        from .preconditions import validate_coherence

        validate_coherence(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect IL-specific metrics."""
        from .metrics import coherence_metrics

        return coherence_metrics(G, node, state_before["dnfr"])

    def _apply_phase_locking(
        self, G: TNFRGraph, node: Any, locking_coefficient: float = 0.3
    ) -> None:
        """Align node phase with neighborhood mean.

        Parameters
        ----------
        locking_coefficient : float
            Phase alignment strength a (default 0.3).

        Notes
        -----
        **Canonical Specification:**

        Steps:
        1. Network phase = circular mean of neighbors
        2. delta = shortest angular diff
        3. theta_new = theta + a*delta
        4. normalize to [0,2*pi]

        **Circular Statistics:**

        Circular mean via complex exponential averaging.

        Handles wrap-around (e.g., 0.1 & 6.2 -> ~0).

        **Telemetry:**

        Telemetry stored in G.graph['IL_phase_locking'].

        **Special Cases:**

        - No neighbors: Phase unchanged (no network to align with)
        - Single neighbor: Aligns toward that neighbor's phase
        - Isolated node: No-op (returns immediately)

        See Also
        --------
        metrics.phase_coherence.compute_phase_alignment: phase alignment
        """
        from ..alias import set_attr
        from ..constants.aliases import ALIAS_THETA

        # Get current node phase
        theta_node = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))

        # Get neighbor phases
        neighbors = list(G.neighbors(node))
        if not neighbors:
            return  # No neighbors, no phase locking

        theta_neighbors = [
            float(get_attr(G.nodes[n], ALIAS_THETA, 0.0))
            for n in neighbors
        ]

        # Compute mean phase using circular mean (angles wrap around 2π)
        # Convert to complex exponentials for circular averaging
        np = get_numpy()

        if np is not None:
            # NumPy vectorized computation
            theta_array = np.array(theta_neighbors)
            complex_phases = np.exp(1j * theta_array)
            mean_complex = np.mean(complex_phases)
            theta_network = np.angle(mean_complex)  # Returns value in [-π, π]

            # Ensure positive phase [0, 2π]
            if theta_network < 0:
                theta_network = float(theta_network + 2 * np.pi)
            else:
                theta_network = float(theta_network)

            # Compute phase difference (considering wrap-around)
            delta_theta = theta_network - theta_node

            # Normalize to [-π, π] for shortest angular distance
            if delta_theta > np.pi:
                delta_theta -= 2 * np.pi
            elif delta_theta < -np.pi:
                delta_theta += 2 * np.pi
            delta_theta = float(delta_theta)

            # Apply phase locking: move θ toward network mean
            theta_new = theta_node + locking_coefficient * delta_theta

            # Normalize to [0, 2π]
            theta_new = float(theta_new % (2 * np.pi))
            import cmath
            import math

            # Convert phases to complex exponentials
            complex_phases = [
                cmath.exp(1j * theta) for theta in theta_neighbors
            ]

            # Compute mean complex phasor
            mean_real = sum(z.real for z in complex_phases) / len(
                complex_phases
            )
            mean_imag = sum(z.imag for z in complex_phases) / len(
                complex_phases
            )
            mean_complex = complex(mean_real, mean_imag)

            # Extract angle (in [-π, π])
            theta_network = cmath.phase(mean_complex)

            # Ensure positive phase [0, 2π]
            if theta_network < 0:
                theta_network += 2 * math.pi

            # Compute phase difference (considering wrap-around)
            delta_theta = theta_network - theta_node

            # Normalize to [-π, π] for shortest angular distance
            if delta_theta > math.pi:
                delta_theta -= 2 * math.pi
            elif delta_theta < -math.pi:
                delta_theta += 2 * math.pi

            # Apply phase locking: move θ toward network mean
            theta_new = theta_node + locking_coefficient * delta_theta

            # Normalize to [0, 2π]
            theta_new = theta_new % (2 * math.pi)

        # Update node phase
        set_attr(G.nodes[node], ALIAS_THETA, theta_new)

        # Store phase locking telemetry
        if "IL_phase_locking" not in G.graph:
            G.graph["IL_phase_locking"] = []

        G.graph["IL_phase_locking"].append(
            {
                "node": node,
                "theta_before": theta_node,
                "theta_after": theta_new,
                "theta_network": theta_network,
                "delta_theta": delta_theta,
                "alignment_achieved": abs(delta_theta)
                * (1 - locking_coefficient),
            }
        )
