"""TNFR Grammar: Grammar Context

Runtime context for grammar validation and operator application tracking.

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

from typing import Any

from .grammar_types import GrammarConfigurationError

# ============================================================================
# Grammar Context
# ============================================================================


class GrammarContext:
    """Context object for grammar validation.

    Minimal implementation for import compatibility.

    Attributes
    ----------
    G : TNFRGraph
        Graph being validated
    cfg_soft : dict
        Soft configuration parameters
    cfg_canon : dict
        Canonical configuration parameters
    norms : dict
        Normalization parameters
    """

    def __init__(
        self,
        G,  # TNFRGraph
        cfg_soft: dict[str, Any] | None = None,
        cfg_canon: dict[str, Any] | None = None,
        norms: dict[str, Any] | None = None,
    ):
        self.G = G
        self.cfg_soft = cfg_soft or {}
        self.cfg_canon = cfg_canon or {}
        self.norms = norms or {}

    @classmethod
    def from_graph(cls, G):  # TNFRGraph
        """Create context from graph.

        Parameters
        ----------
        G : TNFRGraph
            Graph to create context from

        Returns
        -------
        GrammarContext
            New context instance with defaults copied
            
        Raises
        ------
        GrammarConfigurationError
            If TNFR_GRAMMAR_VALIDATE=1 and configuration is invalid
        """
        from ..constants import DEFAULTS
        import copy
        import os

        # Extract configs from graph if present, otherwise use defaults
        cfg_soft = G.graph.get("GRAMMAR", {})
        cfg_canon = G.graph.get("GRAMMAR_CANON", {})
        
        # If empty or missing configs, use defaults
        if not cfg_soft:
            cfg_soft = copy.deepcopy(DEFAULTS.get("GRAMMAR", {}))
        if not cfg_canon:
            cfg_canon = copy.deepcopy(DEFAULTS.get("GRAMMAR_CANON", {}))
            
        # Validate configurations if validation is enabled
        if os.getenv("TNFR_GRAMMAR_VALIDATE") == "1":
            cls._validate_configs(cfg_soft, cfg_canon)
            
        return cls(G, cfg_soft=cfg_soft, cfg_canon=cfg_canon)
        
    @staticmethod
    def _validate_configs(cfg_soft, cfg_canon):
        """Validate configuration dictionaries.
        
        Parameters
        ----------
        cfg_soft : dict
            Soft configuration parameters
        cfg_canon : dict  
            Canonical configuration parameters
            
        Raises
        ------
        GrammarConfigurationError
            If configuration is invalid
        """
        errors = []
        
        # Validate cfg_soft
        if not isinstance(cfg_soft, dict):
            errors.append("cfg_soft must be a mapping/dictionary")
        else:
            # Validate window parameter
            if "window" in cfg_soft:
                window = cfg_soft["window"]
                if not isinstance(window, int) or window < 0:
                    errors.append("cfg_soft.window must be a non-negative integer")
        
        # Validate cfg_canon  
        if not isinstance(cfg_canon, dict):
            errors.append("cfg_canon must be a mapping/dictionary")
        else:
            # Validate thol length constraints
            if ("thol_min_len" in cfg_canon and 
                "thol_max_len" in cfg_canon):
                min_len = cfg_canon["thol_min_len"]
                max_len = cfg_canon["thol_max_len"]
                if (isinstance(min_len, (int, float)) and 
                    isinstance(max_len, (int, float)) and
                    min_len > max_len):
                    errors.append(
                        "cfg_canon.thol_min_len must not exceed thol_max_len"
                    )
            
        if errors:
            # Determine section based on error content
            if any("cfg_soft" in err for err in errors):
                section = "cfg_soft"
            elif any("cfg_canon" in err for err in errors):
                section = "cfg_canon" 
            else:
                section = "configuration"
            raise GrammarConfigurationError(
                section=section,
                messages=errors,
                details=[]
            )


