"""
helpers.py — TNFR canónica

Mejora 1: incorpora el cálculo del Índice de sentido Si por nodo y su escritura
estándar en el grafo (G.nodes[n]["Si"]).

Definición operacional (TNFR):
    Si_i(t) = α·\hat{νf}_i + β·(1 - mean d_φ(i, vec(i))) + γ·(1 - |ΔNFR_i|/max |ΔNFR|)

Donde:
- \hat{νf}_i ∈ [0,1] es la frecuencia estructural normalizada del nodo i.
- d_φ es la distancia de fase envuelta en [0,1] (0 = misma fase, 1 = opuesta).
- ΔNFR_i es el gradiente nodal (magnitud de necesidad de reorganización).
- α, β, γ ≥ 0 y α+β+γ = 1 (si no, se re-normalizan internamente).

Compatibilidad:
- Soporta múltiples alias de atributos para evitar romper proyectos existentes:
  νf:  ["νf", "nu_f", "nu-f", "nu", "freq", "frequency"]
  θ:   ["θ", "theta", "fase", "phi", "phase"]
  ΔNFR:["ΔNFR", "delta_nfr", "dnfr", "gradiente", "grad"]
- El resultado se guarda en "Si" y también puede acumularse en "Si_hist" (lista).

Notas de implementación:
- No se asume una librería de estilos ni unidades específicas; el rango resultante
  es [0,1].
- Si faltan datos, se degradan con defaults prudentes (p. ej., sin vecinos → 0.5
  de sincronía; ΔNFR no disponible → 0 para el término correspondiente).
- Puede leer pesos desde G.graph["Si_weights"] = {"alpha":…, "beta":…, "gamma":…}.
- Si se define G.graph["compute_delta_nfr"], se invoca para rellenar ΔNFR ausentes.

Autor: TNFR | Teoría de la naturaleza fractal resonante
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple, Any, Optional
import math

try:  # opcional, pero recomendado en el proyecto TNFR
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # permitirá importar el módulo aunque no esté networkx

# ----------------------------
# Utilidades de normalización
# ----------------------------

def _min_max_norm(values: Iterable[float], *, eps: float = 1e-12) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 1.0
    vmin, vmax = min(vals), max(vals)
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=eps):
        # Evitamos división por cero; devolvemos un rango mínimo estable
        return float(vmin), float(vmin + 1.0)
    return float(vmin), float(vmax)


def _safe_div(num: float, den: float, *, eps: float = 1e-12) -> float:
    if abs(den) <= eps:
        return 0.0
    return num / den


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# ----------------------------
# Mapas de alias canónicos
# ----------------------------

ALIAS_NU_F = ("νf", "nu_f", "nu-f", "nu", "freq", "frequency")
ALIAS_THETA = ("θ", "theta", "fase", "phi", "phase")
ALIAS_DNFR = ("ΔNFR", "delta_nfr", "dnfr", "gradiente", "grad")


def _get_attr(node_dict: Dict[str, Any], aliases: Tuple[str, ...], default: float = 0.0) -> float:
    for k in aliases:
        if k in node_dict:
            v = node_dict[k]
            try:
                return float(v)
            except Exception:
                pass
    return float(default)


def _wrap_angle(x: float) -> float:
    """Envuelve ángulo a [0, 2π). Acepta grados o radianes; si |x| > 2π*3, intenta
    detectar grados (heurística) y convertir a radianes.
    """
    two_pi = 2.0 * math.pi
    ax = float(x)
    if abs(ax) > two_pi * 3.0:  # heurística simple → probablemente en grados
        ax = math.radians(ax)
    ax = ax % two_pi
    return ax


def _phase_distance01(phi_i: float, phi_j: float) -> float:
    """Distancia de fase envuelta y normalizada a [0,1].
    0 = misma fase; 1 = oposición (π rad). Se define como min(|Δ|, 2π-|Δ|)/π.
    """
    pi = math.pi
    i = _wrap_angle(phi_i)
    j = _wrap_angle(phi_j)
    d = abs(i - j)
    two_pi = 2.0 * pi
    d = min(d, two_pi - d)
    return _clamp01(d / pi)


# ---------------------------------
# Cálculo de Si (Índice de sentido)
# ---------------------------------

def ensure_Si_config(G) -> None:
    """Garantiza que existan pesos canónicos en G.graph["Si_weights"].
    Si no existen o no son válidos, define α=0.4, β=0.4, γ=0.2.
    """
    if not hasattr(G, "graph"):
        return
    weights = G.graph.get("Si_weights")
    if not isinstance(weights, dict):
        G.graph["Si_weights"] = {"alpha": 0.4, "beta": 0.4, "gamma": 0.2}
        return
    a = float(weights.get("alpha", 0.4))
    b = float(weights.get("beta", 0.4))
    c = float(weights.get("gamma", 0.2))
    s = a + b + c
    if s <= 0.0:
        a, b, c = 0.4, 0.4, 0.2
        s = 1.0
    # Re-normalizamos a suma 1 por canonicidad
    G.graph["Si_weights"] = {"alpha": a / s, "beta": b / s, "gamma": c / s}


def compute_Si(G, *, inplace: bool = True) -> Dict[Any, float]:
    """Calcula el Índice de sentido Si para cada nodo del grafo G según la TNFR.

    Parámetros:
        G: grafo (idealmente networkx.Graph/DiGraph).
        inplace: si True, escribe Si en G.nodes[n]["Si"] y acumula en "Si_hist".

    Retorna:
        dict: {nodo: Si ∈ [0,1]}.

    Requisitos mínimos por nodo (con alias):
        - νf (frecuencia estructural):   ALIAS_NU_F
        - θ  (fase estructural):         ALIAS_THETA
        - ΔNFR (gradiente nodal):        ALIAS_DNFR  (si falta, intenta G.graph["compute_delta_nfr"]) 
    """
    if nx is None:
        raise RuntimeError("compute_Si requiere networkx instalado")

    ensure_Si_config(G)
    w = G.graph.get("Si_weights", {"alpha": 0.4, "beta": 0.4, "gamma": 0.2})
    alpha = float(w.get("alpha", 0.4))
    beta = float(w.get("beta", 0.4))
    gamma = float(w.get("gamma", 0.2))

    # 1) Recopilamos νf y ΔNFR para normalización global
    nu_vals = []
    dnfr_vals = []
    for n, d in G.nodes(data=True):
        nu_vals.append(_get_attr(d, ALIAS_NU_F, default=0.0))
        dnfr_vals.append(abs(_get_attr(d, ALIAS_DNFR, default=0.0)))

    # Si ΔNFR no está poblado y existe un callback, lo ejecutamos una vez
    if all(v == 0.0 for v in dnfr_vals):
        cb = G.graph.get("compute_delta_nfr")
        if callable(cb):
            cb(G)  # se espera que llene G.nodes[n][alias ΔNFR]
            dnfr_vals = [abs(_get_attr(d, ALIAS_DNFR, default=0.0)) for _, d in G.nodes(data=True)]

    nu_min, nu_max = _min_max_norm(nu_vals)
    dnfr_max = max(dnfr_vals) if dnfr_vals else 0.0

    # 2) Cálculo por nodo
    out: Dict[Any, float] = {}
    for n, d in G.nodes(data=True):
        # (a) νf normalizada
        nu = _get_attr(d, ALIAS_NU_F, default=0.0)
        nu_hat = _safe_div(nu - nu_min, (nu_max - nu_min)) if nu_max > nu_min else 0.0
        nu_hat = _clamp01(nu_hat)

        # (b) sincronía de fase con vecinos: 1 - media distancia
        if G.is_directed():
            neighs = list(G.predecessors(n)) + list(G.successors(n))
        else:
            neighs = list(G.neighbors(n))
        theta_i = _get_attr(d, ALIAS_THETA, default=0.0)
        if neighs:
            dists = []
            for m in neighs:
                dm = G.nodes[m]
                theta_j = _get_attr(dm, ALIAS_THETA, default=0.0)
                d01 = _phase_distance01(theta_i, theta_j)
                dists.append(d01)
            mean_d = sum(dists) / max(len(dists), 1)
            sync = 1.0 - mean_d
        else:
            sync = 0.5  # sin información vecinal, prior neutro
        sync = _clamp01(sync)

        # (c) término de ΔNFR (queremos 1 cuando el gradiente es bajo)
        dnfr = abs(_get_attr(d, ALIAS_DNFR, default=0.0))
        dnfr_term = 1.0 - _clamp01(_safe_div(dnfr, dnfr_max)) if dnfr_max > 0.0 else 1.0

        # (d) combinación convexa
        Si = alpha * nu_hat + beta * sync + gamma * dnfr_term
        Si = _clamp01(Si)

        out[n] = Si

        if inplace:
            d["Si"] = Si
            # histórico opcional
            try:
                hist = d.get("Si_hist")
                if not isinstance(hist, list):
                    d["Si_hist"] = [Si]
                else:
                    hist.append(Si)
            except Exception:
                d["Si_hist"] = [Si]

    # 3) Exponemos resumen global útil para diagnósticos
    if inplace:
        try:
            vals = list(out.values())
            G.graph.setdefault("Si_stats", {})
            G.graph["Si_stats"].update({
                "mean": float(sum(vals) / max(len(vals), 1)),
                "min": float(min(vals) if vals else 0.0),
                "max": float(max(vals) if vals else 0.0),
            })
        except Exception:
            pass

    return out


# ---------------------------------
# Integración sugerida (dinámica)
# ---------------------------------
# En el bucle de dinámica (antes de seleccionar glifo por nodo):
#     from helpers import compute_Si
#     compute_Si(G, inplace=True)
# …y luego el selector puede priorizar/condicionar por G.nodes[n]["Si"].
