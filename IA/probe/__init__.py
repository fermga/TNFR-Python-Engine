"""IA/probe — Phase 1 TNFR Microscope on an open LLM.

Pure measurement layer: load a frozen LLM, project its hidden states onto a
TNFR-compatible graph, and compute the canonical structural fields
(Φ_s, |∇φ|, K_φ, ξ_C) on that graph.

No training, no operator application — see ``PHASE_1_PLAN.md``.
"""

from __future__ import annotations

__all__ = [
    "load_smollm2",
    "hidden_states_to_tnfr_graph",
    "tnfr_telemetry_from_hidden",
]

# Lazy re-exports keep import-time cheap (transformers/torch are heavy).


def __getattr__(name: str):  # pragma: no cover - thin shim
    if name == "load_smollm2":
        from .loader import load_smollm2

        return load_smollm2
    if name == "hidden_states_to_tnfr_graph":
        from .graph import hidden_states_to_tnfr_graph

        return hidden_states_to_tnfr_graph
    if name == "tnfr_telemetry_from_hidden":
        from .telemetry import tnfr_telemetry_from_hidden

        return tnfr_telemetry_from_hidden
    raise AttributeError(name)
