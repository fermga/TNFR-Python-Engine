"""TNFR public API."""
from __future__ import annotations
try:  # pragma: no cover
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("tnfr")
except PackageNotFoundError:  # pragma: no cover
    try:
        import tomllib
        from pathlib import Path

        with (Path(__file__).resolve().parents[2] / "pyproject.toml").open("rb") as f:
            __version__ = tomllib.load(f)["project"]["version"]
    except (OSError, KeyError, ValueError):  # pragma: no cover
        __version__ = "0+unknown"

# Public API re-exports
from .dynamics import step, run, set_delta_nfr_hook, validate_canon
from .ontosim import preparar_red
from .observers import attach_standard_observer, kuramoto_order
from .helpers import compute_coherence
from .gamma import GAMMA_REGISTRY, eval_gamma, kuramoto_R_psi
from .grammar import enforce_canonical_grammar, on_applied_glyph, apply_glyph_with_grammar
from .sense import (
    glyph_angle,
    glyph_unit,
    sigma_vector_node,
    sigma_vector_from_graph,
    sigma_vector,
    sigma_vector_global,
    push_sigma_snapshot,
    sigma_series,
    sigma_rose,
    register_sigma_callback,
)
from .constants_glyphs import GLYPHS_CANONICAL
from .metrics import (
    register_metrics_callbacks,
    Tg_global, Tg_by_node,
    latency_series, glyphogram_series,
    glyph_top, glyph_dwell_stats, export_history,
)
from .operators import apply_topological_remesh
from .trace import register_trace, CallbackSpec
from .program import play, seq, block, target, wait, THOL, TARGET, WAIT, basic_canonical_example
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

    "attach_standard_observer", "kuramoto_order", "compute_coherence",
    "GAMMA_REGISTRY", "eval_gamma", "kuramoto_R_psi",
    "enforce_canonical_grammar", "on_applied_glyph",
    "apply_glyph_with_grammar",
    "GLYPHS_CANONICAL", "glyph_angle", "glyph_unit",
    "sigma_vector_node", "sigma_vector_from_graph", "sigma_vector", "sigma_vector_global",
    "push_sigma_snapshot", "sigma_series", "sigma_rose",
    "register_sigma_callback",
    "register_metrics_callbacks",
    "register_trace", "CallbackSpec",
    "Tg_global", "Tg_by_node",
    "latency_series", "glyphogram_series",
    "glyph_top", "glyph_dwell_stats",
    "export_history",
    "apply_topological_remesh",
    "play", "seq", "block", "target", "wait", "THOL", "TARGET", "WAIT",
    "cli_main", "build_graph", "get_preset", "NodeState",
    "basic_canonical_example",
    "create_nfr",
    "Operador", "Emision", "Recepcion", "Coherencia", "Disonancia",
    "Acoplamiento", "Resonancia", "Silencio", "Expansion", "Contraccion",
    "Autoorganizacion", "Mutacion", "Transicion", "Recursividad",
    "OPERADORES", "validate_sequence", "run_sequence",
    "__version__",
]





