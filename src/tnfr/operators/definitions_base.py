"""TNFR Operator Base Class

Base Operator class with common functionality for all structural operators.

**Physics**: All operators derive from nodal equation ∂EPI/∂t = νf · ΔNFR(t)
**Implementation**: Each operator applies structural transformations via glyphs
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..rng import validate_graph_seed
from ..types import Glyph, TNFRGraph
from .registry import OperatorMetaAuto

# Metaclass removed – canonical operator set is immutable (see registry).
# Historical dynamic auto-registration deprecated for TNFR grammar purity.

__all__ = ["Operator"]

_PREPARED_OPERATOR_STATE_KEY = "_prepared_operator_state"


class Operator(metaclass=OperatorMetaAuto):
    """Base class for TNFR structural operators.

    Structural operators (Emission, Reception, Coherence, etc.) expose the
    public API for TNFR transformations. Each operator defines a ``name`` and
    ``glyph`` (AL, EN, IL, etc.). Invoking an instance applies its structural
    change to the target node.
    """

    name: ClassVar[str] = "operator"
    # Canonical base class – dynamic registration disabled
    __register__ = False  # retained only for backward compatibility guards
    glyph: ClassVar[Glyph | None] = None

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply the operator to ``node`` under canonical grammar control.

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes, their coherence telemetry and structural
            operator history.
        node : Any
            Identifier or object representing the target node within ``G``.
        **kw : Any
            Additional keyword arguments forwarded to the grammar layer.
            Supported keys include:
            - ``window``: maximum stored glyph-history length
            - ``validate_preconditions``: toggle precondition checks
            - ``collect_metrics``: toggle metrics collection

        Raises
        ------
        NotImplementedError
            If ``glyph`` is :data:`None`, meaning the operator has not been
            bound to a structural symbol.

        Notes
        -----
        Preconditions and incremental grammar selection run before subclass
        execution. If grammar selects a fallback, that canonical operator's
        workflow runs, including its own metadata and metrics. Subclasses
        extend ``_execute``; public calls retain the ``None`` return value.
        Later execution or postcondition failures are not graph transactions.
        """
        if self.glyph is None:
            raise NotImplementedError("Operator without assigned glyph")

        from . import _validated_execution_window
        from .grammar_application import enforce_canonical_grammar
        from .grammar_debt import require_replayable_history
        from .grammar_types import glyph_function_name
        from .registry import get_operator_class

        kw["window"] = _validated_execution_window(G, kw.get("window"))
        validate_graph_seed(G)
        require_replayable_history(G.nodes[node].get("glyph_history"))

        # Hard structural invariants — always enforced before any state
        # mutation, independent of VALIDATE_OPERATOR_PRECONDITIONS.  Coupling
        # and Resonance override this to run the U3 phase gate (Invariant #2).
        self._validate_hard_invariants(G, node)

        self._validate_application_preconditions(G, node, **kw)

        # Select before entering any subclass workflow. A fallback must run
        # its own metadata/metrics and effects, never those of the rejected
        # request (e.g. THOL nesting after an IL replacement).
        selected = enforce_canonical_grammar(
            G,
            node,
            self.glyph,
            kw.get("sequence_context"),
        )
        if glyph_function_name(selected) != glyph_function_name(self.glyph):
            fallback = get_operator_class(glyph_function_name(selected))()
            fallback(G, node, **kw)
            return
        # Subclasses may prepare latency, lineage, regime, or source metadata
        # before delegating to the low-level glyph dispatcher. Resolve the
        # effective branch's active numerical factors here so a rejected
        # factor cannot leave those preparatory writes behind.
        from .factor_contracts import resolve_runtime_operator_factors

        resolve_runtime_operator_factors(
            G.graph.get("GLYPH_FACTORS"), self.glyph, G.graph
        )
        self._execute(G, node, **kw)

    def _validate_application_preconditions(
        self,
        G: TNFRGraph,
        node: Any,
        **kw: Any,
    ) -> None:
        """Validate shared execution controls and configured preconditions."""

        from ._argument_validation import (
            reject_operator_argument,
            require_list_sink,
            validate_common_execution_arguments,
        )

        validate_common_execution_arguments(G.graph, kw, operator=self.name)
        from ._epi_domain import validate_affine_epi_graph_input

        validate_affine_epi_graph_input(
            G,
            node,
            self.glyph,
            operator=self.name,
        )
        collect_metrics = bool(kw.get("collect_metrics", False)) or bool(
            G.graph.get("COLLECT_OPERATOR_METRICS", False)
        )
        if collect_metrics:
            require_list_sink(G.graph, "operator_metrics", operator=self.name)

        monitor = G.graph.get("integrity_monitor")
        if monitor is not None and not all(
            callable(getattr(monitor, method, None))
            for method in ("before_operator", "after_operator")
        ):
            reject_operator_argument(
                self.name,
                "integrity_monitor must provide callable before_operator and "
                "after_operator methods",
            )

        validate_preconditions = kw.get("validate_preconditions", True)
        if validate_preconditions and G.graph.get(
            "VALIDATE_OPERATOR_PRECONDITIONS", False
        ):
            self._validate_preconditions(G, node)

    def _execute(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Execute an already selected operator; subclasses extend this hook.

        Public calls enter through ``__call__`` so argument, hard-invariant,
        precondition and grammar rejection precede subclass metadata writes.
        """
        # Capture state before operator application for metrics and validation
        collect_metrics = kw.get("collect_metrics", False) or G.graph.get(
            "COLLECT_OPERATOR_METRICS", False
        )
        validate_equation = kw.get("validate_nodal_equation", False) or (
            G.graph.get("VALIDATE_NODAL_EQUATION", False)
        )

        state_before = None
        if collect_metrics or validate_equation:
            state_before = self._capture_state(G, node)

        # Structural Integrity Monitor — pre-operator snapshot
        _integrity_monitor = G.graph.get("integrity_monitor")
        if _integrity_monitor is not None:
            _integrity_monitor.before_operator(G, node)

        from .grammar_application import _apply_selected_glyph

        prepared_state = None
        try:
            prepared_state = self._prepare_glyph_application(G, node, **kw)
            if prepared_state is not None and state_before is not None:
                # A pre-operator monitor is permitted to inspect or even alter
                # the graph. Prepared reads therefore own the immediate EN
                # boundary and replace the earlier generic metrics snapshot.
                state_before = self._capture_state(G, node)
                state_before[_PREPARED_OPERATOR_STATE_KEY] = prepared_state
            _apply_selected_glyph(
                G,
                node,
                self.glyph,
                kw.get("window"),
                prepared_state=prepared_state,
            )
            self._after_glyph_application(G, node, **kw)
        except Exception:
            if _integrity_monitor is not None:
                discard_pending = getattr(
                    _integrity_monitor, "discard_pending_operator", None
                )
                if callable(discard_pending):
                    discard_pending()
            raise

        # Structural Integrity Monitor — post-operator evaluation
        # Conservation quality, Lyapunov dE/dt, postconditions, grammar
        if _integrity_monitor is not None:
            _integrity_monitor.after_operator(G, node, self.name)

        # Optional nodal equation validation (∂EPI/∂t = νf · ΔNFR(t))
        if validate_equation and state_before is not None:
            from .nodal_equation import validate_nodal_equation

            dt = float(kw.get("dt", 1.0))  # discrete time step
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

        # Optional metrics collection (capture state after and compute)
        if collect_metrics and state_before is not None:
            metrics = self._collect_metrics(G, node, state_before)
            # Store metrics in graph for retrieval
            if "operator_metrics" not in G.graph:
                G.graph["operator_metrics"] = []
            G.graph["operator_metrics"].append(metrics)

    def _prepare_glyph_application(
        self,
        G: TNFRGraph,
        node: Any,
        **kw: Any,
    ) -> Any:
        """Materialize an optional operator-specific pre-write read set."""

        return None

    def _after_glyph_application(
        self, G: TNFRGraph, node: Any, **kw: Any
    ) -> None:
        """Commit subclass lifecycle state after glyph history is durable.

        The default implementation has no additional lifecycle. Subclasses
        may override this hook when metrics and monitor checks must observe
        metadata derived from the same pre-glyph proposal.
        """

    def _validate_hard_invariants(self, G: TNFRGraph, node: Any) -> None:
        """Validate non-disableable structural invariants.

        Runs on every operator application before any state mutation,
        independent of ``VALIDATE_OPERATOR_PRECONDITIONS``.  Base implementation
        does nothing; Coupling and Resonance override it to enforce the U3
        phase gate (canonical Invariant #2).
        """

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate operator-specific preconditions.

        Override in subclasses to implement specific validation logic.
        Base implementation does nothing.
        """

    def _get_node_attr(self, G: TNFRGraph, node: Any, attr_name: str) -> float:
        """Get node attribute value.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing the node
        node : Any
            Node identifier
        attr_name : str
            Attribute name ("epi", "vf", "dnfr", "theta")

        Returns
        -------
        float
            Attribute value
        """
        alias_map = {
            "epi": ALIAS_EPI,
            "vf": ALIAS_VF,
            "dnfr": ALIAS_DNFR,
            "theta": ALIAS_THETA,
        }

        aliases = alias_map.get(attr_name, (attr_name,))
        return float(get_attr(G.nodes[node], aliases, 0.0))

    def _capture_state(self, G: TNFRGraph, node: Any) -> dict[str, Any]:
        """Capture node state before operator application.

        Returns dict with relevant state for metrics computation.
        """
        return {
            "epi": float(get_attr(G.nodes[node], ALIAS_EPI, 0.0)),
            "vf": float(get_attr(G.nodes[node], ALIAS_VF, 0.0)),
            "dnfr": float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0)),
            "theta": float(get_attr(G.nodes[node], ALIAS_THETA, 0.0)),
        }

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect operator-specific metrics.

        Override in subclasses to implement specific metrics.
        Base implementation returns basic state change.
        """
        # Safely access glyph value
        glyph_value = None
        if self.glyph is not None:
            if hasattr(self.glyph, "value"):
                glyph_value = self.glyph.value
            else:
                glyph_value = str(self.glyph)

        return {
            "operator": self.name,
            "glyph": glyph_value,
            "delta_epi": float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
            - state_before["epi"],
            "delta_vf": float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
            - state_before["vf"],
            "delta_dnfr": float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
            - state_before["dnfr"],
            "delta_theta": float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))
            - state_before["theta"],
        }
