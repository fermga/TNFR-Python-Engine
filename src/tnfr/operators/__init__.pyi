from __future__ import annotations

from typing import Any

from ..node import NodeProtocol
from ..types import Glyph
from .factor_contracts import (
    GLYPH_FACTOR_SPECS as GLYPH_FACTOR_SPECS,
    GLYPH_FACTORS_BY_GLYPH as GLYPH_FACTORS_BY_GLYPH,
    GlyphFactorSpec as GlyphFactorSpec,
    GlyphFactorValidationError as GlyphFactorValidationError,
    canonical_glyph_factor_defaults as canonical_glyph_factor_defaults,
    resolve_operator_factors as resolve_operator_factors,
    resolve_runtime_operator_factors as resolve_runtime_operator_factors,
    runtime_active_glyph_factor_keys as runtime_active_glyph_factor_keys,
    validate_glyph_factor as validate_glyph_factor,
    validate_glyph_factors as validate_glyph_factors,
)
from .word_execution import (
    preflight_network_mutation_sequence as preflight_network_mutation_sequence,
    run_network_sequence as run_network_sequence,
)

Operator: Any
Emission: Any
Reception: Any
Coherence: Any
Dissonance: Any
Coupling: Any
Resonance: Any
Silence: Any
Expansion: Any
Contraction: Any
SelfOrganization: Any
Mutation: Any
Transition: Any
Recursivity: Any
GLYPH_OPERATIONS: Any
JitterCache: Any
JitterCacheManager: Any
OPERATORS: Any
apply_glyph: Any
apply_glyph_obj: Any
apply_network_remesh: Any
apply_remesh_if_globally_stable: Any
apply_topological_remesh: Any
discover_operators: Any

def get_glyph_factors(
    node: NodeProtocol, glyph: Glyph | str | None = ...
) -> dict[str, Any]: ...

get_jitter_manager: Any
get_neighbor_epi: Any
random_jitter: Any
reset_jitter_manager: Any
