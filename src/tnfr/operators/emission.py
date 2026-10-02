"""TNFR Operator: Emission

Emission structural operator (AL) - Foundational activation of nodal resonance.

**Operator contract**: See theory/STRUCTURAL_OPERATORS.md
**Grammar**: theory/UNIFIED_GRAMMAR_RULES.md
"""  # flake8: noqa

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import EMISSION
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Emission(Operator):
    """Emission structural operator (AL).

    Foundational activation of nodal resonance.

    Applies the registered ``AL`` transformation to an existing graph node.
    A zero EPI coordinate does not erase that node's capacity, phase or support.

    TNFR Context
    ------------
    In the Resonant Fractal Nature paradigm, Emission (AL) represents
    the moment when a latent Primary Information Structure (EPI) begins
    to emit coherence toward its surrounding network. This is not passive
    information broadcast but active structural reorganization that boosts
    the node's EPI (the form) from its latent state. Per the canonical
    contract the EPI channel is the direct effect; the existing νf,
    ΔNFR and phase are left untouched.

    **Key Elements:**
    - **Initialization**: A declared emission can activate existing latent form;
      its invocation does not derive spontaneous creation of the nodal substrate
    - **Form Activation**: Raises EPI (Primary Information Structure)
    - **Structural Frequency**: the existing νf is preserved
    - **Network Coupling**: Prepares node for phase alignment
    - **Nodal Equation**: A named EPI jump is distinct from continuous
      ``dEPI/dt = nu_f * DeltaNFR``; execution timing and residuals retain that
      distinction

    **Structural Irreversibility (TNFR.pdf §2.2.1):**
    The implementation retains activation lineage under the historical
    irreversibility contract. This metadata is not a proof of a physically
    irreversible flow, entropy production or emergence of time. Its UTC
    timestamp is provenance, separate from the runtime clock ``graph['_t']``.
    Enclosing transactions may restore graph-owned metadata on failure.
    Successful activation records:

    - **emission_timestamp**: ISO 8601 UTC timestamp of first activation
    - **_emission_activated**: Activation flag retained by this operator
    - **_emission_origin**: Preserved original timestamp (never overwritten)
    - **_structural_lineage**: Genealogical record with:
      - ``origin``: First emission timestamp
      - ``activation_count``: Number of AL applications
      - ``derived_nodes``: list for tracking EPI emergence (future use)
      - ``parent_emission``: Reference to parent node (future use)

    Re-activation increments ``activation_count`` while preserving the
    original timestamp.

    Use Cases
    ---------
    Declared initialization, controlled EPI perturbations and activation-lineage
    tests. Application-domain correspondence requires separate evidence.

    Typical Sequences
    -----------------
    **AL → EN → IL → SHA**: Basic activation with stabilization and silence
    **AL → RA**: Emission with immediate propagation
    **AL → NAV → IL**: Phased activation with transition

    Preconditions
    -------------
    - EPI < the configured ``EPI_LATENT_MAX`` in strict precondition validation
    - Node in latent or low-activation state
    - Sufficient network coupling potential

    Structural Effects
    ------------------
    **EPI**: Increments (form activation) — the direct AL channel
    **νf**: Untouched
    **ΔNFR**: Untouched (AL does not impose reorganization pressure)
    **θ**: Untouched

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
    >>> # Verify recorded activation provenance
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
    Reception : Integrates neighbouring EPI
    """

    __slots__ = ()
    name: ClassVar[str] = EMISSION
    glyph: ClassVar[Glyph] = Glyph.AL

    def _execute(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Validate the shared AL proposal before activation metadata changes.

        This preflight shares lifecycle arithmetic with network stages. The
        ordinary glyph path and its monitor callbacks remain responsible for
        the eventual live write; this is not an arbitrary-callback transaction.
        """
        from datetime import datetime, timezone

        from .al_sha_stage_proposals import (
            commit_emission_lifecycle,
            propose_emission_stage,
        )
        from .factor_contracts import resolve_runtime_operator_factors

        factors = resolve_runtime_operator_factors(
            G.graph.get("GLYPH_FACTORS"), self.glyph, G.graph
        )
        proposal = propose_emission_stage(
            G, node, factors, timestamp=datetime.now(timezone.utc).isoformat()
        )
        commit_emission_lifecycle(G, proposal)
        super()._execute(G, node, **kw)

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate AL-specific preconditions with strict canonical checks.

        Implements TNFR.pdf §2.2.1 precondition validation:
        1. EPI < latent threshold (node in nascent/latent state)
        2. νf >= basal threshold (sufficient structural frequency)
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
