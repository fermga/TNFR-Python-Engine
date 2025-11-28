"""TNFR Operator: Emission

Emission structural operator (AL) - Foundational activation of nodal resonance.

**Physics**: See AGENTS.md § Emission
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""  # flake8: noqa

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import EMISSION
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Emission(Operator):
    """Emission structural operator (AL).

    Foundational activation of nodal resonance.

    Activates structural symbol ``AL`` to initialise outward resonance around a
    nascent node, initiating the first phase of structural reorganization.

    TNFR Context
    ------------
    In the Resonant Fractal Nature paradigm, Emission (AL) represents
    the moment when a latent Primary Information Structure (EPI) begins
    to emit coherence toward its surrounding network. This is not passive
    information broadcast but active structural reorganization that
    increases the node's νf (structural frequency) and initiates positive
    ΔNFR flow.

    **Key Elements:**
        - **Coherent Emergence**: Node exists because it resonates;
            AL starts resonance
    - **Structural Frequency**: Activates νf (Hz_str) to enable reorganization
    - **Network Coupling**: Prepares node for phase alignment
    - **Nodal Equation**: Implements ∂EPI/∂t = νf · ΔNFR(t) with positive ΔNFR

    **Structural Irreversibility (TNFR.pdf §2.2.1):**
    AL is inherently irreversible - once activated, it leaves a persistent
    structural trace that cannot be undone. Each emission marks "time
    zero" for the node and
    establishes genealogical traceability:

    - **emission_timestamp**: ISO 8601 UTC timestamp of first activation
    - **_emission_activated**: Immutable boolean flag
    - **_emission_origin**: Preserved original timestamp (never overwritten)
    - **_structural_lineage**: Genealogical record with:
      - ``origin``: First emission timestamp
      - ``activation_count``: Number of AL applications
      - ``derived_nodes``: List for tracking EPI emergence (future use)
      - ``parent_emission``: Reference to parent node (future use)

    Re-activation increments ``activation_count`` while preserving the
    original timestamp.

    Use Cases
    ---------
    **Biomedical**: HRV coherence training, neural activation, therapy start
    **Cognitive**: Idea germination, learning initiation, creative spark
    **Social**: Team activation, community emergence, ritual initiation

    Typical Sequences
    -----------------
    **AL → EN → IL → SHA**: Basic activation with stabilization and silence
    **AL → RA**: Emission with immediate propagation
    **AL → NAV → IL**: Phased activation with transition

    Preconditions
    -------------
    - EPI < 0.8 (activation threshold)
    - Node in latent or low-activation state
    - Sufficient network coupling potential

    Structural Effects
    ------------------
    **EPI**: Increments (form activation)
    **νf**: Activates/increases (Hz_str)
    **ΔNFR**: Initializes positive reorganization
    **θ**: Influences phase alignment

    Examples
    --------
    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import (
    ...     Emission, Reception, Coherence, Silence
    ... )
    >>> G, node = create_nfr("seed", epi=0.18, vf=1.0)
    >>> run_sequence(
    ...     G,
    ...     node,
    ...     [Emission(), Reception(), Coherence(), Silence()]
    ... )
    >>> # Verify irreversibility
    >>> assert G.nodes[node]["_emission_activated"] is True
    >>> assert "emission_timestamp" in G.nodes[node]
    >>> print(
    ...     f"Activated at: {G.nodes[node]['emission_timestamp']}"
    ... )  # doctest: +SKIP
    Activated at: 2025-11-07T15:47:10.209731+00:00

    See Also
    --------
    Coherence : Stabilizes emitted structures
    Resonance : Propagates emitted coherence
    Reception : Receives external emissions
    """

    __slots__ = ()
    name: ClassVar[str] = EMISSION
    glyph: ClassVar[Glyph] = Glyph.AL

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply AL with structural irreversibility tracking.

        Marks temporal irreversibility before delegating to grammar execution.
        This ensures every emission leaves a persistent structural trace as
        required by TNFR.pdf §2.2.1 (AL - Foundational emission).

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes and structural operator history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Additional keyword arguments forwarded to the grammar layer.
        """
        # Check and clear latency state if reactivating from silence
        self._check_reactivation(G, node)

        # Mark structural irreversibility BEFORE grammar execution
        self._mark_irreversibility(G, node)

        # Delegate to parent __call__ which applies grammar
        super().__call__(G, node, **kw)

    def _check_reactivation(self, G: TNFRGraph, node: Any) -> None:
        """Check and clear latency state when reactivating from silence.

        When AL (Emission) is applied to a node in latent state (from SHA),
        this validates the reactivation and clears the latency attributes.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node.
        node : Any
            Target node being reactivated.

        Warnings
        --------
        - Warns if node is reactivated after extended silence (duration check)
        - Warns if EPI has drifted from preserved value during silence
        """
        if G.nodes[node].get("latent", False):
            # Node is in latent state, reactivating from silence
            silence_duration = G.nodes[node].get("silence_duration", 0.0)

            # Get max silence duration threshold from graph config
            max_silence = G.graph.get("MAX_SILENCE_DURATION", float("inf"))

            # Validate reactivation timing
            if silence_duration > max_silence:
                warnings.warn(
                    f"Node {node} reactivating after extended silence "
                    f"(duration: {silence_duration:.2f}, "
                    f"max: {max_silence:.2f})",
                    stacklevel=3,
                )

            # Check EPI preservation integrity
            preserved_epi = G.nodes[node].get("preserved_epi")
            if preserved_epi is not None:
                # get_attr already imported at module top

                current_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
                epi_drift = abs(current_epi - preserved_epi)

                # Allow small numerical drift (1% tolerance)
                if epi_drift > 0.01 * abs(preserved_epi):
                    warnings.warn(
                        f"Node {node} EPI drifted during silence "
                        f"(preserved: {preserved_epi:.3f}, "
                        f"current: {current_epi:.3f}, "
                        f"drift: {epi_drift:.3f})",
                        stacklevel=3,
                    )

            # Clear latency state
            del G.nodes[node]["latent"]
            if "latency_start_time" in G.nodes[node]:
                del G.nodes[node]["latency_start_time"]
            if "preserved_epi" in G.nodes[node]:
                del G.nodes[node]["preserved_epi"]
            if "silence_duration" in G.nodes[node]:
                del G.nodes[node]["silence_duration"]

    def _mark_irreversibility(self, G: TNFRGraph, node: Any) -> None:
        """Mark structural irreversibility for AL operator.

        According to TNFR.pdf §2.2.1, AL (Emission) is structurally
        irreversible:
        "Una vez activado, AL reorganiza el campo. No puede deshacerse."

        This method establishes:
        - Temporal marker: ISO timestamp of first emission
        - Activation flag: Persistent boolean indicating AL was activated
        - Structural lineage: Genealogical record for EPI traceability

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node.
        node : Any
            Target node for emission marking.

        Notes
        -----
        On first activation:
        - Sets emission_timestamp (ISO format)
        - Sets _emission_activated = True (immutable)
        - Sets _emission_origin (timestamp copy for preservation)
        - Initializes _structural_lineage dict

        On re-activation:
        - Preserves original timestamp
        - Increments activation_count in lineage
        """
        from datetime import datetime, timezone

        from ..alias import set_attr_str
        from ..constants.aliases import ALIAS_EMISSION_TIMESTAMP

        # Check if this is first activation
        if "_emission_activated" not in G.nodes[node]:
            # Generate UTC timestamp in ISO format
            emission_timestamp = datetime.now(timezone.utc).isoformat()

            # Set canonical timestamp using alias system (string values)
            set_attr_str(
                G.nodes[node], ALIAS_EMISSION_TIMESTAMP, emission_timestamp
            )

            # Set persistent activation flag (immutable marker)
            G.nodes[node]["_emission_activated"] = True

            # Preserve origin timestamp (never overwritten)
            G.nodes[node]["_emission_origin"] = emission_timestamp

            # Initialize structural lineage for genealogical traceability
            G.nodes[node]["_structural_lineage"] = {
                "origin": emission_timestamp,
                "activation_count": 1,
                "derived_nodes": [],  # Nodes that emerge from this emission
                "parent_emission": None,  # If derived from another node
            }
        else:
            # Re-activation: increment counter, keep original timestamp
            if "_structural_lineage" in G.nodes[node]:
                G.nodes[node]["_structural_lineage"]["activation_count"] += 1

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate AL-specific preconditions with strict canonical checks.

        Implements TNFR.pdf §2.2.1 precondition validation:
        1. EPI < latent threshold (node in nascent/latent state)
        2. νf > basal threshold (sufficient structural frequency)
        3. Network connectivity check (warning for isolated nodes)

        Raises
        ------
        ValueError
            If EPI too high or νf too low for emission
        """
        from .preconditions.emission import validate_emission_strict

        validate_emission_strict(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect AL-specific metrics."""
        from .metrics import emission_metrics

        return emission_metrics(
            G,
            node,
            state_before["epi"],
            state_before["vf"],
        )
