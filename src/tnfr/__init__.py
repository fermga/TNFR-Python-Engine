
from __future__ import annotations
"""
TNFR — Teoría de la Naturaleza Fractal Resonante
API pública del paquete.

Ecuación nodal:
    ∂EPI/∂t = νf · ΔNFR(t)
"""

__version__ = "4.5.2"

# Re-exports de la API pública
from .dynamics import step, run, set_delta_nfr_hook, validate_canon
from .ontosim import preparar_red
from .observers import attach_standard_observer, coherencia_global, orden_kuramoto
from .gamma import GAMMA_REGISTRY, eval_gamma, kuramoto_R_psi
from .grammar import enforce_canonical_grammar, on_applied_glifo, apply_glyph_with_grammar
from .sense import (
    GLYPHS_CANONICAL, glyph_angle, glyph_unit,
    sigma_vector_node, sigma_vector_global, sigma_vector,
    push_sigma_snapshot, sigma_series, sigma_rose,
    register_sigma_callback,
)
from .metrics import (
    register_metrics_callbacks,
    Tg_global, Tg_by_node,
    latency_series, glifogram_series,
    glyph_top, glyph_dwell_stats, export_history,
)
from .operators import aplicar_remesh_red_topologico
from .trace import register_trace
from .program import play, seq, block, target, wait, THOL, TARGET, WAIT, ejemplo_canonico_basico
from .cli import main as cli_main
from .scenarios import build_graph
from .presets import get_preset
from .types import NodeState
from .structural import (
    create_nfr,
    Operador,
    Emision,
    Recepcion,
    Coherencia,
    Disonancia,
    Acoplamiento,
    Resonancia,
    Silencio,
    Expansion,
    Contraccion,
    Autoorganizacion,
    Mutacion,
    Transicion,
    Recursividad,
    OPERADORES,
    validate_sequence,
    run_sequence,
)


__all__ = [
    "preparar_red",
    "step", "run", "set_delta_nfr_hook", "validate_canon",

    "attach_standard_observer", "coherencia_global", "orden_kuramoto",
    "GAMMA_REGISTRY", "eval_gamma", "kuramoto_R_psi",
    "enforce_canonical_grammar", "on_applied_glifo",
    "apply_glyph_with_grammar",
    "GLYPHS_CANONICAL", "glyph_angle", "glyph_unit",
    "sigma_vector_node", "sigma_vector_global", "sigma_vector",
    "push_sigma_snapshot", "sigma_series", "sigma_rose",
    "register_sigma_callback",
    "register_metrics_callbacks",
    "register_trace",
    "Tg_global", "Tg_by_node",
    "latency_series", "glifogram_series",
    "glyph_top", "glyph_dwell_stats",
    "export_history",
    "aplicar_remesh_red_topologico",
    "play", "seq", "block", "target", "wait", "THOL", "TARGET", "WAIT",
    "cli_main", "build_graph", "get_preset", "NodeState",
    "ejemplo_canonico_basico",
    "create_nfr",
    "Operador", "Emision", "Recepcion", "Coherencia", "Disonancia",
    "Acoplamiento", "Resonancia", "Silencio", "Expansion", "Contraccion",
    "Autoorganizacion", "Mutacion", "Transicion", "Recursividad",
    "OPERADORES", "validate_sequence", "run_sequence",
    "__version__",
]





