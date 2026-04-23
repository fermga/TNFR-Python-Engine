"""Project an LLM hidden-state tensor onto a TNFR-compatible graph.

Conventions match ``tnfr.physics.fields``:
    - Node attribute ``EPI`` ∈ ℝ (we normalise to [0, 1])
    - Node attribute ``theta`` ∈ [-π, π] — the structural phase φ
    - Node attribute ``nu_f`` ∈ ℝ⁺ — structural frequency

Three projection methods are supported (see PHASE_1_PLAN.md §4):
    - "cosine_topk"  : top-k cosine similarity edges
    - "cosine_thr"   : edge if cos(h_i, h_j) > threshold
    - "attention"    : top-k from a supplied attention matrix
"""

from __future__ import annotations

import math
from typing import Any, Literal

import networkx as nx
import numpy as np

ProjectionMethod = Literal["cosine_topk", "cosine_thr", "attention"]


def _to_numpy(h: Any) -> np.ndarray:
    """Accept a torch.Tensor or a numpy array; return float32 numpy."""
    if hasattr(h, "detach"):
        h = h.detach().to("cpu").float().numpy()
    arr = np.asarray(h, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D (T, d) hidden states, got shape {arr.shape}")
    return arr


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return x / norms


def _phase_from_random_projection(h: np.ndarray, seed: int) -> np.ndarray:
    """Project each row onto two fixed random unit vectors and take arg().

    Deterministic given (seed, hidden_dim). Returns θ ∈ [-π, π].
    """
    rng = np.random.default_rng(seed)
    d = h.shape[1]
    w_re = rng.standard_normal(d).astype(np.float32)
    w_im = rng.standard_normal(d).astype(np.float32)
    w_re /= np.linalg.norm(w_re) + 1e-12
    w_im /= np.linalg.norm(w_im) + 1e-12
    re = h @ w_re
    im = h @ w_im
    return np.arctan2(im, re).astype(np.float32)


def _nu_f_from_local_entropy(sims: np.ndarray) -> np.ndarray:
    """Per-row Shannon entropy of softmax-normalised similarities, in [0, 1]."""
    # Numerically stable softmax over each row.
    s = sims - sims.max(axis=1, keepdims=True)
    p = np.exp(s)
    p_sum = p.sum(axis=1, keepdims=True)
    p_sum = np.where(p_sum < 1e-12, 1.0, p_sum)
    p = p / p_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    max_ent = math.log(max(sims.shape[1], 2))
    return (ent / max_ent).astype(np.float32)


def hidden_states_to_tnfr_graph(
    h: Any,
    method: ProjectionMethod = "cosine_topk",
    *,
    k: int = 8,
    threshold: float = 0.5,
    attention: Any | None = None,
    seed: int = 0,
    max_nodes: int = 512,
) -> nx.Graph:
    """Project a hidden-state matrix onto a TNFR graph.

    Parameters
    ----------
    h : torch.Tensor | np.ndarray of shape (T, d) or (1, T, d)
    method : projection strategy.
    k : top-k neighbours (cosine_topk / attention).
    threshold : cosine threshold (cosine_thr).
    attention : optional (T, T) attention matrix (attention method).
    seed : RNG seed for the phase projection (reproducibility).
    max_nodes : truncate at this many tokens to bound memory.

    Returns
    -------
    nx.Graph with per-node attributes ``EPI``, ``theta``, ``nu_f``.
    """
    H = _to_numpy(h)
    if H.shape[0] > max_nodes:
        H = H[:max_nodes]
    T = H.shape[0]

    Hn = _l2_normalize(H)
    sims = Hn @ Hn.T  # (T, T) cosine similarities
    np.fill_diagonal(sims, -np.inf)  # exclude self in top-k / threshold

    G = nx.Graph()

    # Nodes & attributes
    epi_raw = np.linalg.norm(H, axis=1)
    epi_max = float(epi_raw.max()) if epi_raw.size else 1.0
    epi = (epi_raw / epi_max).astype(np.float32) if epi_max > 0 else epi_raw
    theta = _phase_from_random_projection(H, seed=seed)
    nu_f = _nu_f_from_local_entropy(sims if T > 1 else np.zeros((T, 1)))

    for i in range(T):
        G.add_node(
            i,
            EPI=float(epi[i]),
            theta=float(theta[i]),
            nu_f=float(nu_f[i]),
        )

    # Edges
    if method == "cosine_topk":
        if T > 1:
            kk = min(k, T - 1)
            idx = np.argpartition(-sims, kth=kk - 1, axis=1)[:, :kk]
            for i in range(T):
                for j in idx[i]:
                    if i == j:
                        continue
                    w = float(sims[i, j])
                    if not math.isfinite(w):
                        continue
                    G.add_edge(int(i), int(j), weight=w)
    elif method == "cosine_thr":
        iu, ju = np.where(sims > threshold)
        for i, j in zip(iu, ju):
            if i < j:
                G.add_edge(int(i), int(j), weight=float(sims[i, j]))
    elif method == "attention":
        if attention is None:
            raise ValueError("method='attention' requires `attention` matrix")
        A = _to_numpy(attention)
        if A.shape[0] > max_nodes:
            A = A[:max_nodes, :max_nodes]
        np.fill_diagonal(A, -np.inf)
        kk = min(k, T - 1)
        idx = np.argpartition(-A, kth=kk - 1, axis=1)[:, :kk]
        for i in range(T):
            for j in idx[i]:
                if i == j:
                    continue
                w = float(A[i, j])
                if not math.isfinite(w):
                    continue
                G.add_edge(int(i), int(j), weight=w)
    else:  # pragma: no cover - typing guard
        raise ValueError(f"unknown method: {method!r}")

    return G
