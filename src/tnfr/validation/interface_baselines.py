"""Classical graph-local baselines for structural-interface benchmarks.

This module is the *fair-comparison* layer for TNFR Structural Interface Theory.
It provides classical, well-understood, graph-local node scores that any TNFR
interface claim must be compared against.  The closest classical analogue to the
TNFR phase-gate stress is :func:`local_disagreement`; the remaining baselines
(graph total variation, local class entropy, label-propagation residual, graph
cut contribution, neighbour distance, degree, feature deviation, and
constant/random controls) widen the comparison so that any reported TNFR
advantage is measured against strong, not only weak, references.

Design rules
------------
- Pure-Python and deterministic.  ``random_baseline`` uses a seeded
  :class:`random.Random` over a stable node order; everything else is exact.
- Read-only: no graph attribute is mutated.
- Each baseline returns a ``dict[node, float]`` so it can be ranked uniformly by
  :func:`tnfr.validation.structural_interface.evaluate_interface_scores`.
- Formulas are documented per function (Milestone 2 acceptance criterion).

Honest note
-----------
For a *binary* label encoded as a phase (0 vs ``π``), graph total variation on
the phase signal is proportional to :func:`local_disagreement`.  This is stated
explicitly rather than hidden: the two baselines coincide up to a scale factor
when the only per-node signal is the label itself.

References
----------
- ``docs/STRUCTURAL_INTERFACE_THEORY_PLAN.md`` §"Fair benchmark design"
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency guard
    nx = None  # type: ignore[assignment]

__all__ = [
    "local_disagreement",
    "graph_total_variation",
    "local_class_entropy",
    "label_propagation_residual",
    "graph_cut_contribution",
    "mean_neighbour_distance",
    "degree_score",
    "feature_deviation",
    "constant_baseline",
    "random_baseline",
    "compute_all_baselines",
    "BASELINE_FORMULAS",
]

_DEFAULT_DISTANCE_KEY = "distance"

#: Short human-readable formula notes, surfaced in reports for transparency.
BASELINE_FORMULAS: Mapping[str, str] = {
    "local_disagreement": "count of neighbours whose state differs from the node",
    "graph_total_variation": "sum_j |v_i - v_j| over incident edges (v = numeric signal or label code)",
    "local_class_entropy": "Shannon entropy of class counts over the closed neighbourhood, normalized by log(#classes)",
    "label_propagation_residual": "1 - f_i[own_class] after clamped label propagation (alpha, iterations)",
    "graph_cut_contribution": "sum over cross-class incident edges of similarity weight 1/(1+distance)",
    "mean_neighbour_distance": "mean feature-space distance to neighbours (edge distance attribute)",
    "degree": "node degree",
    "feature_deviation": "|x_i - mean(x)| / std(x) for a chosen numeric feature",
    "constant": "constant 1.0 for every node (control)",
    "random": "uniform[0,1) per node from a seeded RNG (control)",
}


def _require_networkx() -> None:
    if nx is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("networkx is required for interface baselines")


def _stable_nodes(G: Any) -> list[Any]:
    """Return graph nodes in a deterministic order (by ``repr``)."""
    return sorted(G.nodes(), key=lambda node: repr(node))


def _has_numeric_attr(G: Any, key: str) -> bool:
    for node in G.nodes():
        value = G.nodes[node].get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
    return G.number_of_nodes() > 0


# ---------------------------------------------------------------------------
# Boundary-sensitive baselines
# ---------------------------------------------------------------------------


def local_disagreement(G: Any, *, state_key: str) -> dict[Any, float]:
    """Count, per node, the neighbours whose ``state_key`` differs.

    Formula: ``score_i = |{ j ~ i : state_j != state_i }|``.

    This is the closest classical analogue to TNFR phase-gate violations and is
    the primary reference baseline for any interface claim.
    """
    _require_networkx()
    scores: dict[Any, float] = {}
    for node in G.nodes():
        own = G.nodes[node].get(state_key)
        scores[node] = float(
            sum(
                1
                for neighbour in G.neighbors(node)
                if G.nodes[neighbour].get(state_key) != own
            )
        )
    return scores


def _state_codes(G: Any, state_key: str) -> dict[Any, int]:
    classes = sorted(
        {G.nodes[node].get(state_key) for node in G.nodes()}, key=lambda c: repr(c)
    )
    return {cls: code for code, cls in enumerate(classes)}


def graph_total_variation(
    G: Any,
    *,
    value_key: str | None = None,
    state_key: str | None = None,
) -> dict[Any, float]:
    """Per-node graph total variation of a numeric signal.

    Formula: ``score_i = sum_{j ~ i} |v_i - v_j|``.

    When ``value_key`` is given, ``v`` is that numeric node attribute.  Otherwise
    ``state_key`` must be given and categories are mapped to integer codes
    (``v`` = code).  For a binary label this is proportional to
    :func:`local_disagreement`.
    """
    _require_networkx()
    if value_key is None and state_key is None:
        raise ValueError("provide value_key or state_key")

    if value_key is not None:
        value = {node: float(G.nodes[node].get(value_key, 0.0)) for node in G.nodes()}
    else:
        codes = _state_codes(G, state_key)  # type: ignore[arg-type]
        value = {
            node: float(codes.get(G.nodes[node].get(state_key), 0))
            for node in G.nodes()
        }

    scores: dict[Any, float] = {}
    for node in G.nodes():
        own = value[node]
        scores[node] = float(
            sum(abs(own - value[neighbour]) for neighbour in G.neighbors(node))
        )
    return scores


def local_class_entropy(
    G: Any, *, state_key: str, normalize: bool = True
) -> dict[Any, float]:
    """Shannon entropy of class counts over each closed neighbourhood.

    The closed neighbourhood of ``i`` is ``{i} ∪ N(i)``.  Higher entropy means a
    more mixed neighbourhood, i.e. a stronger interface.  When ``normalize`` is
    True the entropy is divided by ``log(#global_classes)`` so scores lie in
    ``[0, 1]``.
    """
    _require_networkx()
    global_classes = {G.nodes[node].get(state_key) for node in G.nodes()}
    norm = math.log(len(global_classes)) if len(global_classes) > 1 else 0.0

    scores: dict[Any, float] = {}
    for node in G.nodes():
        counts: dict[Any, int] = {}
        members = [node, *G.neighbors(node)]
        for member in members:
            cls = G.nodes[member].get(state_key)
            counts[cls] = counts.get(cls, 0) + 1
        total = sum(counts.values())
        entropy = 0.0
        if total > 0:
            for count in counts.values():
                p = count / total
                entropy -= p * math.log(p)
        if normalize and norm > 0:
            scores[node] = float(entropy / norm)
        else:
            scores[node] = float(entropy)
    return scores


def label_propagation_residual(
    G: Any,
    *,
    state_key: str,
    alpha: float = 0.85,
    iterations: int = 30,
) -> dict[Any, float]:
    """Clamped label-propagation disagreement residual.

    Each node starts with a one-hot class vector.  At every iteration the soft
    label is updated as ``f_i <- (1 - alpha) * seed_i + alpha * mean_j f_j``.
    After ``iterations`` steps the residual is ``1 - f_i[own_class]`` (the soft
    probability mass that propagation moved away from the node's own class).
    Boundary nodes accumulate larger residuals.  Deterministic.
    """
    _require_networkx()
    nodes = list(G.nodes())
    classes = sorted(
        {G.nodes[node].get(state_key) for node in nodes}, key=lambda c: repr(c)
    )
    if len(classes) <= 1:
        return {node: 0.0 for node in nodes}
    index = {cls: i for i, cls in enumerate(classes)}
    k = len(classes)

    seed = {node: [0.0] * k for node in nodes}
    for node in nodes:
        seed[node][index[G.nodes[node].get(state_key)]] = 1.0
    f = {node: list(seed[node]) for node in nodes}

    for _ in range(max(1, int(iterations))):
        updated: dict[Any, list[float]] = {}
        for node in nodes:
            neighbours = list(G.neighbors(node))
            if neighbours:
                acc = [0.0] * k
                for neighbour in neighbours:
                    fj = f[neighbour]
                    for c in range(k):
                        acc[c] += fj[c]
                inv = 1.0 / len(neighbours)
                propagated = [value * inv for value in acc]
            else:
                propagated = list(seed[node])
            updated[node] = [
                (1.0 - alpha) * seed[node][c] + alpha * propagated[c]
                for c in range(k)
            ]
        f = updated

    residual: dict[Any, float] = {}
    for node in nodes:
        own = index[G.nodes[node].get(state_key)]
        total = sum(f[node]) or 1.0
        residual[node] = float(1.0 - f[node][own] / total)
    return residual


def graph_cut_contribution(
    G: Any,
    *,
    state_key: str,
    distance_key: str = _DEFAULT_DISTANCE_KEY,
) -> dict[Any, float]:
    """Per-node contribution to a similarity-weighted class cut.

    Formula: ``score_i = sum_{j ~ i, state_j != state_i} 1 / (1 + distance_ij)``.

    Unlike :func:`local_disagreement` (a raw count), this weights cross-class
    edges by feature-space proximity, so a node that is *close* to a
    differently-labelled neighbour scores higher than one whose cross-class
    neighbour is far.
    """
    _require_networkx()
    scores: dict[Any, float] = {}
    for node in G.nodes():
        own = G.nodes[node].get(state_key)
        total = 0.0
        for neighbour in G.neighbors(node):
            if G.nodes[neighbour].get(state_key) != own:
                distance = float(G.edges[node, neighbour].get(distance_key, 0.0))
                total += 1.0 / (1.0 + distance)
        scores[node] = float(total)
    return scores


# ---------------------------------------------------------------------------
# Topology / feature / control baselines
# ---------------------------------------------------------------------------


def mean_neighbour_distance(
    G: Any, *, distance_key: str = _DEFAULT_DISTANCE_KEY
) -> dict[Any, float]:
    """Mean feature-space distance to neighbours (edge ``distance_key``)."""
    _require_networkx()
    scores: dict[Any, float] = {}
    for node in G.nodes():
        distances = [
            float(G.edges[node, neighbour].get(distance_key, 0.0))
            for neighbour in G.neighbors(node)
        ]
        scores[node] = sum(distances) / len(distances) if distances else 0.0
    return scores


def degree_score(G: Any) -> dict[Any, float]:
    """Node degree (topology-only control)."""
    _require_networkx()
    return {node: float(G.degree[node]) for node in G.nodes()}


def feature_deviation(G: Any, *, value_key: str) -> dict[Any, float]:
    """Standardized absolute deviation of a numeric feature from its mean.

    Formula: ``score_i = |x_i - mean(x)| / std(x)`` (std clamped to 1 if zero).
    A simple, domain-agnostic feature baseline.
    """
    _require_networkx()
    values = {node: float(G.nodes[node].get(value_key, 0.0)) for node in G.nodes()}
    if not values:
        return {}
    mean = sum(values.values()) / len(values)
    variance = sum((v - mean) ** 2 for v in values.values()) / len(values)
    std = math.sqrt(variance) or 1.0
    return {node: abs(value - mean) / std for node, value in values.items()}


def constant_baseline(G: Any, *, value: float = 1.0) -> dict[Any, float]:
    """Constant score for every node (trivial control)."""
    _require_networkx()
    return {node: float(value) for node in G.nodes()}


def random_baseline(G: Any, *, seed: int = 0) -> dict[Any, float]:
    """Deterministic uniform[0,1) score per node from a seeded RNG."""
    _require_networkx()
    rng = random.Random(int(seed))
    return {node: rng.random() for node in _stable_nodes(G)}


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def compute_all_baselines(
    G: Any,
    *,
    state_key: str,
    distance_key: str = _DEFAULT_DISTANCE_KEY,
    phase_key: str = "phase",
    feature_key: str | None = None,
    seed: int = 0,
) -> dict[str, dict[Any, float]]:
    """Compute the full classical baseline suite as named node-score maps.

    The graph-total-variation baseline uses the numeric ``phase_key`` signal when
    every node carries it; otherwise it falls back to state-code total variation
    (documented to coincide with :func:`local_disagreement` up to scaling for
    binary labels).  ``feature_key`` adds :func:`feature_deviation` when supplied.
    """
    _require_networkx()
    use_phase = _has_numeric_attr(G, phase_key)
    maps: dict[str, dict[Any, float]] = {
        "local_disagreement": local_disagreement(G, state_key=state_key),
        "graph_total_variation": graph_total_variation(
            G,
            value_key=phase_key if use_phase else None,
            state_key=None if use_phase else state_key,
        ),
        "local_class_entropy": local_class_entropy(G, state_key=state_key),
        "label_propagation_residual": label_propagation_residual(
            G, state_key=state_key
        ),
        "graph_cut_contribution": graph_cut_contribution(
            G, state_key=state_key, distance_key=distance_key
        ),
        "mean_neighbour_distance": mean_neighbour_distance(
            G, distance_key=distance_key
        ),
        "degree": degree_score(G),
        "constant": constant_baseline(G),
        "random": random_baseline(G, seed=seed),
    }
    if feature_key is not None:
        maps["feature_deviation"] = feature_deviation(G, value_key=feature_key)
    return maps
