"""TNFR Operator: Transition

Transition structural operator (NAV) - Controlled regime handoff.

**Physics**: See AGENTS.md § Transition
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

import math
import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import TRANSITION
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator
from .registry import register_operator


class Transition(Operator):
    """Transition structural operator (NAV) - Controlled regime handoff.

    Activates glyph ``NAV`` to guide the node through a controlled transition between
    structural regimes, managing hand-offs across states.

    TNFR Context: Transition (NAV) manages movement between coherence regimes with minimal
    disruption. NAV adjusts θ, νf, and ΔNFR to navigate thresholds smoothly, preventing
    collapse during regime shifts. Essential for change management.

    Use Cases: State transitions, regime changes, threshold crossings, transformation
    processes, managed evolution.

    Typical Sequences: AL → NAV → IL (activate-transition-stabilize), NAV → ZHIR (transition
    enables mutation), SHA → NAV → AL (silence-transition-reactivation), IL → NAV → OZ
    (stable-transition-explore).

    Versatility: NAV is highly compatible with most operators as transition manager.

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, THETA_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Transition
    >>> G, node = create_nfr("mu", vf=0.85, theta=0.40)
    >>> ramps = iter([(0.12, -0.25)])
    >>> def handoff(graph):
    ...     d_vf, d_theta = next(ramps)
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.nodes[node][THETA_PRIMARY] += d_theta
    ...     graph.nodes[node][DNFR_PRIMARY] = abs(d_vf) * 0.5
    >>> set_delta_nfr_hook(G, handoff)
    >>> run_sequence(G, node, [Transition()])
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    0.97
    >>> round(G.nodes[node][THETA_PRIMARY], 2)
    0.15
    >>> round(G.nodes[node][DNFR_PRIMARY], 2)
    0.06

    **Biomedical**: Sleep stage transitions, developmental phases, recovery processes
    **Cognitive**: Learning phase transitions, attention shifts, mode switching
    **Social**: Organizational change, cultural transitions, leadership handoffs
    """

    __slots__ = ()
    name: ClassVar[str] = TRANSITION
    glyph: ClassVar[Glyph] = Glyph.NAV

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply NAV with regime detection and controlled transition.

        Implements TNFR.pdf §2.3.11 canonical transition logic:
        1. Detect current structural regime (latent/active/resonant)
        2. Handle latency reactivation if node was in silence (SHA → NAV)
        3. Apply grammar and structural transformation
        4. Collect metrics (if enabled)

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes and structural operator history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Additional keyword arguments:
            - phase_shift (float): Override default phase shift per regime
            - vf_factor (float): Override νf scaling for active regime (default: 1.0)
            - Other args forwarded to grammar layer

        Notes
        -----
        Regime-specific transformations (TNFR.pdf §2.3.11):

        **Latent → Active** (νf < 0.05 or latent flag):
        - νf × 1.2 (20% increase for gradual reactivation)
        - θ + 0.1 rad (small phase shift)
        - ΔNFR × 0.7 (30% reduction for smooth transition)

        **Active** (baseline state):
        - νf × vf_factor (default 1.0, configurable)
        - θ + 0.2 rad (standard phase shift)
        - ΔNFR × 0.8 (20% reduction)

        **Resonant → Active** (EPI > 0.5 AND νf > 0.8):
        - νf × 0.95 (5% reduction for stability)
        - θ + 0.15 rad (careful phase shift)
        - ΔNFR × 0.9 (10% reduction, gentle)

        Telemetry stored in G.graph["_nav_transitions"] tracks:
        - regime_origin, vf_before/after, theta_before/after, dnfr_before/after
        """
        from ..alias import get_attr
        from ..constants.aliases import ALIAS_EPI

        # 1. Detect current regime and store for metrics collection
        current_regime = self._detect_regime(G, node)
        G.nodes[node]["_regime_before"] = current_regime

        # 2. Handle latency reactivation if applicable
        if G.nodes[node].get("latent", False):
            self._handle_latency_transition(G, node)

        # 3. Validate preconditions (if enabled)
        validate_preconditions = kw.get("validate_preconditions", True) or G.graph.get(
            "VALIDATE_PRECONDITIONS", False
        )
        if validate_preconditions:
            self._validate_preconditions(G, node)

        # 4. Capture state before for metrics/validation
        collect_metrics = kw.get("collect_metrics", False) or G.graph.get(
            "COLLECT_OPERATOR_METRICS", False
        )
        validate_equation = kw.get("validate_nodal_equation", False) or G.graph.get(
            "VALIDATE_NODAL_EQUATION", False
        )

        state_before = None
        if collect_metrics or validate_equation:
            state_before = self._capture_state(G, node)

        # 5. Apply grammar
        from . import apply_glyph_with_grammar

        apply_glyph_with_grammar(G, [node], self.glyph, kw.get("window"))

        # 6. Execute structural transition (BEFORE metrics collection)
        self._apply_structural_transition(G, node, current_regime, **kw)

        # 7. Optional nodal equation validation
        if validate_equation and state_before is not None:
            from .nodal_equation import validate_nodal_equation

            dt = float(kw.get("dt", 1.0))
            strict = G.graph.get("NODAL_EQUATION_STRICT", False)
            epi_after = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))

            validate_nodal_equation(
                G,
                node,
                epi_before=state_before["epi"],
                epi_after=epi_after,
                dt=dt,
                operator_name=self.name,
                strict=strict,
            )

        # 8. Optional metrics collection (AFTER structural transformation)
        if collect_metrics and state_before is not None:
            metrics = self._collect_metrics(G, node, state_before)
            if "operator_metrics" not in G.graph:
                G.graph["operator_metrics"] = []
            G.graph["operator_metrics"].append(metrics)

    def _detect_regime(self, G: TNFRGraph, node: Any) -> str:
        """Detect current structural regime: latent/active/resonant.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node.
        node : Any
            Target node.

        Returns
        -------
        str
            Regime classification: "latent", "active", or "resonant"

        Notes
        -----
        Classification criteria:
        - **Latent**: latent flag set OR νf < 0.05 (minimal reorganization capacity)
        - **Resonant**: EPI > 0.5 AND νf > 0.8 (high form + high frequency)
        - **Active**: Default (baseline operational state)
        """
        from ..alias import get_attr
        from ..constants.aliases import ALIAS_EPI, ALIAS_VF

        epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
        latent = G.nodes[node].get("latent", False)

        if latent or vf < 0.05:
            return "latent"
        elif epi > 0.5 and vf > 0.8:
            return "resonant"
        else:
            return "active"

    def _handle_latency_transition(self, G: TNFRGraph, node: Any) -> None:
        """Handle transition from latent state (SHA → NAV flow).

        Similar to Emission._check_reactivation but for NAV-specific transitions.
        Validates silence duration and clears latency attributes.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node.
        node : Any
            Target node being reactivated.

        Warnings
        --------
        - Warns if node transitioning after extended silence (duration > MAX_SILENCE_DURATION)
        - Warns if EPI drifted significantly during silence (> 1% tolerance)

        Notes
        -----
        Clears latency-related attributes:
        - latent (flag)
        - latency_start_time (ISO timestamp)
        - preserved_epi (EPI snapshot from SHA)
        - silence_duration (computed duration)
        """
        from datetime import datetime, timezone

        # Verify silence duration if timestamp available
        if "latency_start_time" in G.nodes[node]:
            start = datetime.fromisoformat(G.nodes[node]["latency_start_time"])
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            G.nodes[node]["silence_duration"] = duration

            max_silence = G.graph.get("MAX_SILENCE_DURATION", float("inf"))
            if duration > max_silence:
                warnings.warn(
                    f"Node {node} transitioning after extended silence "
                    f"(duration: {duration:.2f}s, max: {max_silence:.2f}s)",
                    stacklevel=4,
                )

        # Check EPI preservation integrity
        preserved_epi = G.nodes[node].get("preserved_epi")
        if preserved_epi is not None:
            from ..alias import get_attr
            from ..constants.aliases import ALIAS_EPI

            current_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
            epi_drift = abs(current_epi - preserved_epi)

            # Allow small numerical drift (1% tolerance)
            if epi_drift > 0.01 * abs(preserved_epi):
                warnings.warn(
                    f"Node {node} EPI drifted during silence "
                    f"(preserved: {preserved_epi:.3f}, current: {current_epi:.3f}, "
                    f"drift: {epi_drift:.3f})",
                    stacklevel=4,
                )

        # Clear latency state
        del G.nodes[node]["latent"]
        if "latency_start_time" in G.nodes[node]:
            del G.nodes[node]["latency_start_time"]
        if "preserved_epi" in G.nodes[node]:
            del G.nodes[node]["preserved_epi"]
        # Keep silence_duration for telemetry/metrics - don't delete it

    def _apply_structural_transition(self, G: TNFRGraph, node: Any, regime: str, **kw: Any) -> None:
        """Apply structural transformation based on regime origin.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node.
        node : Any
            Target node.
        regime : str
            Origin regime: "latent", "active", or "resonant"
        **kw : Any
            Optional overrides:
            - phase_shift (float): Custom phase shift
            - vf_factor (float): Custom νf scaling for active regime

        Notes
        -----
        Applies regime-specific transformations to θ, νf, and ΔNFR following
        TNFR.pdf §2.3.11. All changes use canonical alias system (set_attr)
        to ensure proper attribute resolution.

        Telemetry appended to G.graph["_nav_transitions"] for analysis.
        """
        from ..alias import get_attr, set_attr
        from ..constants.aliases import ALIAS_DNFR, ALIAS_THETA, ALIAS_VF

        # Get current state
        theta = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))
        vf = float(get_attr(G.nodes[node], ALIAS_VF, 1.0))
        dnfr = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

        # Apply regime-specific adjustments
        if regime == "latent":
            # Latent → Active: gradual reactivation
            vf_new = vf * 1.2  # 20% increase
            theta_shift = kw.get("phase_shift", 0.1)  # Small phase shift
            theta_new = (theta + theta_shift) % (2 * math.pi)
            dnfr_new = dnfr * 0.7  # 30% reduction for smooth transition
        elif regime == "active":
            # Active: standard transition
            vf_new = vf * kw.get("vf_factor", 1.0)  # Configurable
            theta_shift = kw.get("phase_shift", 0.2)  # Standard shift
            theta_new = (theta + theta_shift) % (2 * math.pi)
            dnfr_new = dnfr * 0.8  # 20% reduction
        else:  # resonant
            # Resonant → Active: careful transition (high energy state)
            vf_new = vf * 0.95  # 5% reduction for stability
            theta_shift = kw.get("phase_shift", 0.15)  # Careful phase shift
            theta_new = (theta + theta_shift) % (2 * math.pi)
            dnfr_new = dnfr * 0.9  # 10% reduction, gentle

        # Apply changes via canonical alias system
        set_attr(G.nodes[node], ALIAS_VF, vf_new)
        set_attr(G.nodes[node], ALIAS_THETA, theta_new)
        set_attr(G.nodes[node], ALIAS_DNFR, dnfr_new)

        # Telemetry tracking
        if "_nav_transitions" not in G.graph:
            G.graph["_nav_transitions"] = []
        G.graph["_nav_transitions"].append(
            {
                "node": node,
                "regime_origin": regime,
                "vf_before": vf,
                "vf_after": vf_new,
                "theta_before": theta,
                "theta_after": theta_new,
                "dnfr_before": dnfr,
                "dnfr_after": dnfr_new,
                "phase_shift": theta_new - theta,
            }
        )

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate NAV-specific preconditions."""
        from .preconditions import validate_transition

        validate_transition(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect NAV-specific metrics."""
        from .metrics import transition_metrics

        return transition_metrics(
            G,
            node,
            state_before["dnfr"],
            state_before["vf"],
            state_before["theta"],
            epi_before=state_before.get("epi"),
        )