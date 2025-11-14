"""Definitions for canonical TNFR structural operators.

Structural operators (Emission, Reception, Coherence, etc.) are the public-facing
API for applying TNFR transformations to nodes. Each operator is associated with
a specific glyph (structural symbol like AL, EN, IL, etc.) that represents the
underlying transformation.

English identifiers are the public API. Spanish wrappers were removed in
TNFR 2.0, so downstream code must import these classes directly.

**Physics & Theory References:**
- Complete operator physics: AGENTS.md § Canonical Operators
- Grammar constraints (U1-U6): UNIFIED_GRAMMAR_RULES.md
- Nodal equation (∂EPI/∂t = νf · ΔNFR): AGENTS.md § Foundational Physics

**Implementation:**
- Canonical grammar validation: src/tnfr/operators/grammar.py
- Operator registry: src/tnfr/operators/registry.py

**Note**: This is a facade module. Individual operators are implemented in
separate files for better maintainability. All imports preserve backward
compatibility.
"""

from __future__ import annotations

from .definitions_base import Operator
from .emission import Emission
from .reception import Reception
from .coherence import Coherence
from .dissonance import Dissonance
from .coupling import Coupling
from .resonance import Resonance
from .silence import Silence
from .expansion import Expansion
from .contraction import Contraction
from .self_organization import SelfOrganization
from .mutation import Mutation
from .transition import Transition
from .recursivity import Recursivity
from .introspection import (
    OperatorMeta,
    OPERATOR_METADATA,
    get_operator_meta,
    iter_operator_meta,
)
from .grammar_error_factory import (
    ExtendedGrammarError,
    collect_grammar_errors,
    make_grammar_error,
)

__all__ = [
    "Operator",
    "Emission",
    "Reception",
    "Coherence",
    "Dissonance",
    "Coupling",
    "Resonance",
    "Silence",
    "Expansion",
    "Contraction",
    "SelfOrganization",
    "Mutation",
    "Transition",
    "Recursivity",
    # Introspection exports
    "OperatorMeta",
    "OPERATOR_METADATA",
    "get_operator_meta",
    "iter_operator_meta",
    "ExtendedGrammarError",
    "collect_grammar_errors",
    "make_grammar_error",
]
