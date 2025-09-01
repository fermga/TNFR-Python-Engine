"""Registro de gammas."""
from __future__ import annotations
from typing import Dict, Any, Tuple
import math
import cmath
import logging

from .constants import ALIAS_THETA
from .helpers import get_attr


logger = logging.getLogger(__name__)



def _ensure_kuramoto_cache(G, t) -> None:
    """Cachea (R, ψ) en ``G.graph`` para el paso ``t`` actual.

    El cálculo se invalida si cambia el paso o la firma de los nodos.
    """
    nodes_sig = (len(G), hash(tuple(G)))
    cache = G.graph.get("_kuramoto_cache")
    if (
        cache is None
        or cache.get("t") != t
        or cache.get("nodes_sig") != nodes_sig
    ):
        R, psi = kuramoto_R_psi(G)
        G.graph["_kuramoto_cache"] = {
            "t": t,
            "nodes_sig": nodes_sig,
            "R": R,
            "psi": psi,
        }


def kuramoto_R_psi(G) -> Tuple[float, float]:
    """Devuelve (R, ψ) del orden de Kuramoto usando θ de todos los nodos."""
    acc = 0 + 0j
    n = 0
    for node in G.nodes():
        nd = G.nodes[node]
        th = get_attr(nd, ALIAS_THETA, 0.0)
        acc += cmath.exp(1j * th)
        n += 1
    if n == 0:
        return 0.0, 0.0
    z = acc / n
    return abs(z), math.atan2(z.imag, z.real)


def _kuramoto_common(G, node, cfg):
    """Return ``(θ_i, R, ψ)`` for Kuramoto-based γ functions.

    Lee los valores cacheados de orden global ``R`` y fase media ``ψ`` y
    obtiene la fase del nodo ``θ_i``. ``cfg`` se acepta solo para mantener una
    firma homogénea con los evaluadores de ``Γ``.
    """
    cache = G.graph.get("_kuramoto_cache", {})
    R = float(cache.get("R", 0.0))
    psi = float(cache.get("psi", 0.0))
    th_i = get_attr(G.nodes[node], ALIAS_THETA, 0.0)
    return th_i, R, psi


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
    th_i, R, psi = _kuramoto_common(G, node, cfg)
    return beta * (R - R0) * math.cos(th_i - psi)


def gamma_kuramoto_bandpass(G, node, t, cfg: Dict[str, Any]) -> float:
    """Γ = β · R(1-R) · sign(cos(θ_i - ψ))"""
    beta = float(cfg.get("beta", 0.0))
    th_i, R, psi = _kuramoto_common(G, node, cfg)
    sgn = 1.0 if math.cos(th_i - psi) >= 0.0 else -1.0
    return beta * R * (1.0 - R) * sgn


def gamma_kuramoto_tanh(G, node, t, cfg: Dict[str, Any]) -> float:
    """Acoplamiento saturante tipo tanh para Γi(R).

    Fórmula: Γ = β · tanh(k·(R - R0)) · cos(θ_i - ψ)
      - β: ganancia del acoplamiento
      - k: pendiente de la tanh (cuán rápido satura)
      - R0: umbral de activación
    """
    beta = float(cfg.get("beta", 0.0))
    k = float(cfg.get("k", 1.0))
    R0 = float(cfg.get("R0", 0.0))
    th_i, R, psi = _kuramoto_common(G, node, cfg)
    return beta * math.tanh(k * (R - R0)) * math.cos(th_i - psi)


def gamma_harmonic(G, node, t, cfg: Dict[str, Any]) -> float:
    """Forzamiento armónico coherente con el campo global de fase.

    Fórmula: Γ = β · sin(ω·t + φ) · cos(θ_i - ψ)
      - β: ganancia del acoplamiento
      - ω: frecuencia angular del forzante
      - φ: fase inicial del forzante
    """
    beta = float(cfg.get("beta", 0.0))
    omega = float(cfg.get("omega", 1.0))
    phi = float(cfg.get("phi", 0.0))
    th_i, _, psi = _kuramoto_common(G, node, cfg)
    return beta * math.sin(omega * t + phi) * math.cos(th_i - psi)


# ``GAMMA_REGISTRY`` asocia el nombre del acoplamiento con un par
# ``(fn, needs_kuramoto)`` donde ``fn`` es la función evaluadora y
# ``needs_kuramoto`` indica si requiere precomputar el orden global de fase.
GAMMA_REGISTRY = {
    "none": (gamma_none, False),
    "kuramoto_linear": (gamma_kuramoto_linear, True),
    "kuramoto_bandpass": (gamma_kuramoto_bandpass, True),
    "kuramoto_tanh": (gamma_kuramoto_tanh, True),
    "harmonic": (gamma_harmonic, True),
}


def eval_gamma(G, node, t, *, strict: bool = False) -> float:
    """Evalúa Γi para `node` según la especificación en G.graph['GAMMA'].

    Si ``strict`` es ``True`` las excepciones encontradas durante la
    evaluación se reelevarán en lugar de devolver ``0.0``.
    """
    spec = G.graph.get("GAMMA", {"type": "none"})
    fn, needs_kuramoto = GAMMA_REGISTRY.get(
        spec.get("type", "none"), (gamma_none, False)
    )
    if needs_kuramoto:
        _ensure_kuramoto_cache(G, t)
    try:
        return float(fn(G, node, t, spec))
    except (KeyError, TypeError, ValueError):
        logger.exception("Fallo al evaluar Γi para nodo %s en t=%s", node, t)
        if strict:
            raise
        return 0.0
