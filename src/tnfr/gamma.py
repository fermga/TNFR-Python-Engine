"""gamma.py — TNFR canónica

Γi(R): acoplamientos de red para la ecuación nodal extendida
    ∂EPI/∂t = νf · ΔNFR(t) + Γi(R)

Provee:
- kuramoto_R_psi(G): (R, ψ) orden de Kuramoto en la red
- GAMMA_REGISTRY: registro de acoplamientos canónicos
- eval_gamma(G, node, t): evalúa Γ para cada nodo según G.graph['GAMMA']
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import math
import cmath

from .constants import ALIAS_THETA


def _get_attr(nd: Dict[str, Any], aliases, default: float = 0.0) -> float:
    """Obtiene el primer atributo presente en nd según aliases."""
    for k in aliases:
        if k in nd:
            return nd[k]
    return default


def kuramoto_R_psi(G) -> Tuple[float, float]:
    """Devuelve (R, ψ) del orden de Kuramoto usando θ de todos los nodos."""
    acc = 0 + 0j
    n = 0
    for node in G.nodes():
        nd = G.nodes[node]
        th = _get_attr(nd, ALIAS_THETA, 0.0)
        acc += cmath.exp(1j * th)
        n += 1
    if n == 0:
        return 0.0, 0.0
    z = acc / n
    return abs(z), math.atan2(z.imag, z.real)


# -----------------
# Γi(R) canónicos
# -----------------


def gamma_none(G, node, t, cfg: Dict[str, Any]) -> float:
    return 0.0


def gamma_kuramoto_linear(G, node, t, cfg: Dict[str, Any]) -> float:
    """Acoplamiento lineal de Kuramoto para Γi(R).

    Fórmula: Γ = β · (R - R0) · cos(θ_i - ψ)
      - R ∈ [0,1] es el orden global de fase.
      - ψ es la fase media (dirección de coordinación).
      - β, R0 son parámetros (ganancia/umbral).

    Uso: refuerza integración cuando la red ya exhibe coherencia de fase (R>R0).
    """
    beta = float(cfg.get("beta", 0.0))
    R0 = float(cfg.get("R0", 0.0))
    R, psi = kuramoto_R_psi(G)
    th_i = _get_attr(G.nodes[node], ALIAS_THETA, 0.0)
    return beta * (R - R0) * math.cos(th_i - psi)


def gamma_kuramoto_bandpass(G, node, t, cfg: Dict[str, Any]) -> float:
    """Γ = β · R(1-R) · sign(cos(θ_i - ψ))"""
    beta = float(cfg.get("beta", 0.0))
    R, psi = kuramoto_R_psi(G)
    th_i = _get_attr(G.nodes[node], ALIAS_THETA, 0.0)
    sgn = 1.0 if math.cos(th_i - psi) >= 0.0 else -1.0
    return beta * R * (1.0 - R) * sgn


GAMMA_REGISTRY = {
    "none": gamma_none,
    "kuramoto_linear": gamma_kuramoto_linear,
    "kuramoto_bandpass": gamma_kuramoto_bandpass,
}


def eval_gamma(G, node, t) -> float:
    """Evalúa Γi para `node` según la especificación en G.graph['GAMMA']."""
    spec = G.graph.get("GAMMA", {"type": "none"})
    fn = GAMMA_REGISTRY.get(spec.get("type", "none"), gamma_none)
    try:
        return float(fn(G, node, t, spec))
    except Exception:
        return 0.0
