"""TNFR Operator Base Class

Base Operator class with common functionality for all structural operators.

**Physics**: All operators derive from nodal equation ∂EPI/∂t = νf · ΔNFR(t)
**Implementation**: Each operator applies structural transformations via glyphs
"""

from __future__ import annotations

import math
import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..types import Glyph, TNFRGraph
from .registry import OperatorMetaAuto
from ..utils import get_numpy  # noqa: F401 (compatibility)

# Metaclass removed – canonical operator set is immutable (see registry).
# Historical dynamic auto-registration deprecated for TNFR grammar purity.

__all__ = ["Operator"]

# T'HOL canonical bifurcation constants
_THOL_SUB_EPI_SCALING = 0.25  # Sub-EPI ~25% of parent (first-order)
_THOL_EMERGENCE_CONTRIBUTION = 0.1  # Parent EPI +10% of sub-EPI


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
            - ``window``: constrain grammar window
            - ``validate_preconditions``: toggle precondition checks
            - ``collect_metrics``: toggle metrics collection

        Raises
        ------
        NotImplementedError
            If ``glyph`` is :data:`None`, meaning the operator has not been
            bound to a structural symbol.

        Notes
        -----
        The invocation delegates to
        :func:`tnfr.validation.apply_glyph_with_grammar`, which enforces
        the TNFR grammar before activating the structural transformation. The
        grammar may expand, contract or stabilise the neighbourhood so that the
        operator preserves canonical closure and coherence.
        """
        if self.glyph is None:
            raise NotImplementedError("Operator without assigned glyph")

        # Optional precondition validation
        validate_preconditions = kw.get("validate_preconditions", True)
        if validate_preconditions and G.graph.get(
            "VALIDATE_OPERATOR_PRECONDITIONS", False
        ):
            self._validate_preconditions(G, node)

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

        from . import apply_glyph_with_grammar

        apply_glyph_with_grammar(G, [node], self.glyph, kw.get("window"))

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

