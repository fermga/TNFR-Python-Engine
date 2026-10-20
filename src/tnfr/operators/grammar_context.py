"""TNFR Grammar: Grammar Context

Runtime context for grammar validation and operator application tracking.

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

from typing import Any, Mapping

from ..config.parsing import parse_bool
from .grammar_types import GrammarConfigurationError


def _normalize_sequence_context(
    context: Mapping[str, Any] | None, *, compatibility_profile: str | None = None
) -> dict[str, Any]:
    """Copy admitted word context, defaulting to the shared core contracts.

    Legacy adjacency and THOL-terminal preferences require explicit selection.
    The explicit argument retains precedence over a context profile.
    """
    if context is None:
        context = {}
    if not isinstance(context, Mapping):
        raise TypeError("sequence context must be a mapping")
    normalized = dict(context)
    for name in ("initial_epi_nonzero", "diagnostic"):
        if name in normalized:
            normalized[name] = parse_bool(normalized[name])
    profile = (
        normalized.get("compatibility_profile", "core")
        if compatibility_profile is None
        else compatibility_profile
    )
    if type(profile) is not str:
        raise TypeError("compatibility_profile must be 'core' or 'legacy'")
    if profile not in {"core", "legacy"}:
        raise ValueError("compatibility_profile must be 'core' or 'legacy'")
    normalized["compatibility_profile"] = profile
    return normalized


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
        cfg_soft: Mapping[str, Any] | None = None,
        cfg_canon: Mapping[str, Any] | None = None,
        norms: Mapping[str, Any] | None = None,
    ):
        self.G = G
        self.cfg_soft = _context_mapping(cfg_soft, "cfg_soft")
        self.cfg_canon = _context_mapping(cfg_canon, "cfg_canon")
        self.norms = _context_mapping(norms, "norms")

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
        import copy
        import os

        from ..constants import DEFAULTS

        # Extract configs from graph if present, otherwise use defaults
        cfg_soft = G.graph.get("GRAMMAR", {})
        cfg_canon = G.graph.get("GRAMMAR_CANON", {})
        if not isinstance(cfg_soft, Mapping):
            raise TypeError("GRAMMAR must be a mapping")
        if not isinstance(cfg_canon, Mapping):
            raise TypeError("GRAMMAR_CANON must be a mapping")

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
        from ..validation.window import validate_window

        errors = []

        # Validate cfg_soft
        if not isinstance(cfg_soft, Mapping):
            errors.append("cfg_soft must be a mapping/dictionary")
        else:
            # Validate window parameter
            if "window" in cfg_soft:
                try:
                    validate_window(cfg_soft["window"])
                except (TypeError, ValueError):
                    errors.append("cfg_soft.window must be a non-negative integer")

        # Validate cfg_canon
        if not isinstance(cfg_canon, Mapping):
            errors.append("cfg_canon must be a mapping/dictionary")

        if errors:
            # Determine section based on error content
            if any("cfg_soft" in err for err in errors):
                section = "cfg_soft"
            elif any("cfg_canon" in err for err in errors):
                section = "cfg_canon"
            else:
                section = "configuration"
            raise GrammarConfigurationError(
                section=section, messages=errors, details=[]
            )


def _context_mapping(value: Mapping[str, Any] | None, label: str) -> Mapping[str, Any]:
    """Keep optional absence distinct from malformed supplied configuration.

    Supplied mappings remain live references; this is not a frozen policy.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping or None")
    return value
