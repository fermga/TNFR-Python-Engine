"""Compute TNFR tetrad fields on a graph projected from LLM hidden states.

Thin wrapper around ``tnfr.physics.fields.compute_unified_telemetry``. No new
physics — Phase 1 is pure measurement.
"""

from __future__ import annotations

from typing import Any, Iterable

from .graph import ProjectionMethod, hidden_states_to_tnfr_graph


def tnfr_telemetry_from_hidden(
    hidden_states: Iterable[Any],
    *,
    layers: Iterable[int],
    method: ProjectionMethod = "cosine_topk",
    k: int = 8,
    seed: int = 0,
    max_nodes: int = 512,
) -> dict[int, dict[str, Any]]:
    """Run the TNFR microscope over a list of layer hidden states.

    Parameters
    ----------
    hidden_states : sequence indexable by layer index. Each entry is a
        tensor / array of shape (T, d) or (1, T, d). Typically the
        ``hidden_states`` tuple returned by a HuggingFace causal LM
        (length = n_layers + 1, including the embedding layer at index 0).
    layers : layer indices to probe (e.g. (4, 12, 23)).
    method, k, seed, max_nodes : forwarded to ``hidden_states_to_tnfr_graph``.

    Returns
    -------
    dict[layer_index, telemetry_dict] where each telemetry_dict is the
    output of ``tnfr.physics.fields.compute_unified_telemetry``.
    """
    from tnfr.physics.fields import compute_unified_telemetry

    out: dict[int, dict[str, Any]] = {}
    for layer in layers:
        h = hidden_states[layer]
        G = hidden_states_to_tnfr_graph(
            h, method=method, k=k, seed=seed, max_nodes=max_nodes
        )
        out[int(layer)] = compute_unified_telemetry(G)
    return out
