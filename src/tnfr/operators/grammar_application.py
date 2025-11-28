"""TNFR Grammar: Grammar Application

Functions for applying operators with grammar enforcement at runtime.

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import Glyph
else:
    from ..types import Glyph

from .grammar_context import GrammarContext
from .grammar_patterns import recognize_il_sequences


def apply_glyph_with_grammar(
    G,  # TNFRGraph
    nodes: Any,
    glyph: Any,
    window: Any = None,
) -> None:
    """Apply glyph to nodes with grammar validation.

    Applies the specified glyph to each node in the iterable using the canonical
    TNFR operator implementation.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing nodes
    nodes : Any
        Node, list of nodes, or node iterable to apply glyph to
    glyph : Any
        Glyph to apply
    window : Any, optional
        Grammar window constraint

    Notes
    -----
    This function delegates to apply_glyph for each node, which wraps
    the node in NodeNX and applies the glyph operation.
    """
    from . import apply_glyph

    # Handle single node or iterable of nodes
    # Check if it's a single hashable node or an iterable
    try:
        # Try to treat as single hashable node
        hash(nodes)
        # If hashable, it's a single node
        nodes_iter = [nodes]
    except (TypeError, AttributeError):
        # Not hashable, treat as iterable
        # Convert to list to allow multiple iterations if needed
        try:
            nodes_iter = list(nodes)
        except TypeError:
            # If not iterable, wrap in list
            nodes_iter = [nodes]

    for node in nodes_iter:
        apply_glyph(G, node, glyph, window=window)
        
        # Check for IL sequences in node history after applying glyph
        if "glyph_history" in G.nodes[node]:
            history = G.nodes[node]["glyph_history"]
            if len(history) >= 2:
                # Check last two glyphs for canonical patterns
                # Convert to list to support slicing
                history_list = list(history)
                
                # Convert string names to Glyphs for recognition
                glyph_history = []
                for item in history_list[-2:]:
                    if isinstance(item, str):
                        if item.startswith('Glyph.'):
                            # Handle 'Glyph.AL' format
                            glyph_name = item.split('.')[1]
                            try:
                                glyph_history.append(Glyph[glyph_name])
                            except KeyError:
                                glyph_history.append(item)
                        else:
                            # Handle direct glyph name 'IL'
                            try:
                                glyph_history.append(Glyph[item])
                            except KeyError:
                                glyph_history.append(item)
                    else:
                        glyph_history.append(item)
                        
                recognized = recognize_il_sequences(glyph_history)
                
                if recognized:
                    # Initialize graph-level pattern tracking if needed
                    if "recognized_coherence_patterns" not in G.graph:
                        G.graph["recognized_coherence_patterns"] = []
                    
                    # Add recognized patterns to graph tracking
                    for pattern in recognized:
                        pattern_info = {
                            "node": node,
                            "pattern_name": pattern["pattern_name"],
                            "position": len(history) - 2 + pattern["position"],
                            "is_antipattern": pattern.get(
                                "is_antipattern", False
                            ),
                        }
                        G.graph["recognized_coherence_patterns"].append(
                            pattern_info
                        )
                        
                        # Emit warnings for antipatterns if not already done
                        is_antipattern = pattern.get("is_antipattern", False)
                        severity = pattern.get("severity", "")
                        if (is_antipattern and
                                severity in ("warning", "error")):
                            import warnings
                            pattern_name = pattern["pattern_name"]
                            warnings.warn(
                                f"Anti-pattern detected: {pattern_name}",
                                UserWarning
                            )


def on_applied_glyph(G, n, applied: Any) -> None:  # G: TNFRGraph, n: NodeId
    """Record glyph application in node history.

    Minimal stub for tracking operator sequences.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing node
    n : NodeId
        Node identifier
    applied : Any
        Applied glyph or operator name
    """
    # Minimal stub for telemetry
    if "glyph_history" not in G.nodes[n]:
        G.nodes[n]["glyph_history"] = []
    G.nodes[n]["glyph_history"].append(applied)


def enforce_canonical_grammar(
    G,  # TNFRGraph
    n,  # NodeId
    cand: Any,
    ctx: Any = None,
) -> Any:
    """Minimal stub for backward compatibility.

    This function is a no-op stub maintained for compatibility with existing
    code that expects this interface. It simply returns the candidate as-is.

    For actual grammar validation, use validate_grammar() from unified_grammar.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing node
    n : NodeId
        Node identifier
    cand : Any
        Candidate glyph/operator
    ctx : Any, optional
        Grammar context (ignored)

    Returns
    -------
    Any
        The candidate unchanged
        
    Raises
    ------
    GrammarConfigurationError
        If TNFR_GRAMMAR_VALIDATE=1 and graph configuration is invalid
    """
    import os
    
    # Validate configuration if validation is enabled  
    if os.getenv("TNFR_GRAMMAR_VALIDATE") == "1":
        # Create context to trigger validation
        GrammarContext.from_graph(G)
        
    return cand


