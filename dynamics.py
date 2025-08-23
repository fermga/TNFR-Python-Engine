"""
dynamics.py — TNFR canónica

Mejora 1: integración del Índice de sentido (Si) en el bucle de dinámica
para sesgar/condicionar la selección glífica antes de aplicar transformadores
sobre los nodos. La ecuación nodal se mantiene:

    ∂EPI/∂t = νf · ΔNFR(t)

Cambios claves:
- Se importa y ejecuta helpers.compute_Si(G, inplace=True) en cada paso.
- Selector glífico con sesgo por Si (alto→I’L/estabilización; bajo→O’Z/NA’V).
- Interfaz abierta: si el usuario define G.graph["glyph_selector"], se utiliza
  ese callback; en caso contrario se usa el selector canónico de este archivo.
- Callbacks opcionales en G.graph para extender:
    • compute_delta_nfr(G): escribe ΔNFR por nodo if faltante.
    • after_step(G, k): hook post-paso.

Compatibilidad:
- Alias de atributos: νf/nu_f, θ/theta, ΔNFR/delta_nfr… (ver ALIAS_*).
- Si no están los operadores correspondientes, los glifos degradan a no-op.

Autor: TNFR | Teoría de la naturaleza fractal resonante
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Iterable
import math

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None

from helpers import compute_Si  # Mejora 1

# Operadores (glifos). Cada función debe aceptar (G, n, **kw) y mutar el nodo.
try:
    import operators as OPS  # type: ignore
except Exception:  # pragma: no cover
    OPS = None

# -----------------------------
# Alias canónicos de atributos
# -----------------------------
ALIAS_NU_F = ("νf", "nu_f", "nu-f", "nu", "freq", "frequency")
ALIAS_THETA = ("θ", "theta", "fase", "phi", "phase")
ALIAS_DNFR = ("ΔNFR", "delta_nfr", "dnfr", "gradiente", "grad")
ALIAS_EPI = ("EPI", "PSI", "psi", "epi")


def _get_attr(node_dict: Dict[str, Any], aliases: Iterable[str], default: float = 0.0) -> float:
    for k in aliases:
        if k in node_dict:
            try:
                return float(node_dict[k])
            except Exception:
                pass
    return float(default)


def _set_attr(node_dict: Dict[str, Any], aliases: Iterable[str], value: float) -> None:
    # Escribimos en el primer alias canónico
    key = next(iter(aliases))
    node_dict[key] = float(value)


# --------------------------------
# Ecuación nodal (actualización EPI)
# --------------------------------

def update_epi_via_nodal_equation(G, *, dt: float = 1.0) -> None:
    """Aplica ∂EPI/∂t = νf·ΔNFR por nodo con integración de Euler simple.
    Guarda dEPI_dt y EPI_prev para diagnósticos.
    """
    if nx is None:
        raise RuntimeError("Se requiere networkx para dynamics.update_epi_via_nodal_equation")

    for n, d in G.nodes(data=True):
        nu = _get_attr(d, ALIAS_NU_F, default=0.0)
        dnfr = _get_attr(d, ALIAS_DNFR, default=0.0)
        epi = _get_attr(d, ALIAS_EPI, default=0.0)
        dEPI_dt = nu * dnfr
        d["dEPI_dt"] = dEPI_dt
        d["EPI_prev"] = epi
        epi_next = epi + dEPI_dt * dt
        _set_attr(d, ALIAS_EPI, epi_next)


# -------------------
# ΔNFR por defecto
# -------------------

def default_compute_delta_nfr(G) -> None:
    """Compute ΔNFR si falta, mezclando dos componentes sencillas:
    - Diferencia media de fase con vecinos (0..1, donde 1 = oposición de fase)
    - Gradiente local de EPI
    Resultado en [0, 1] aprox. (no estrictamente acotado si EPI crece mucho).
    Si existe G.graph["compute_delta_nfr"], se usa ese callback en su lugar.
    """
    cb = G.graph.get("compute_delta_nfr")
    if callable(cb):
        cb(G)
        return

    for n, d in G.nodes(data=True):
        if any(k in d for k in ALIAS_DNFR):
            continue  # respetamos valores ya presentes
        theta_i = _get_attr(d, ALIAS_THETA, default=0.0)
        epi_i = _get_attr(d, ALIAS_EPI, default=0.0)
        if G.is_directed():
            neighs = list(G.predecessors(n)) + list(G.successors(n))
        else:
            neighs = list(G.neighbors(n))
        if not neighs:
            _set_attr(d, ALIAS_DNFR, 0.0)
            continue
        # fase: distancia envuelta normalizada (0..1)
        def phase_dist(a: float, b: float) -> float:
            two_pi = 2.0 * math.pi
            def wrap(x: float) -> float:
                if abs(x) > two_pi * 3.0:
                    x = math.radians(x)
                return x % two_pi
            i, j = wrap(a), wrap(b)
            dd = abs(i - j)
            dd = min(dd, two_pi - dd)
            return dd / math.pi  # 0..1
        dists = []
        grad_epi = []
        for m in neighs:
            dm = G.nodes[m]
            dists.append(phase_dist(theta_i, _get_attr(dm, ALIAS_THETA, 0.0)))
            grad_epi.append(abs(epi_i - _get_attr(dm, ALIAS_EPI, 0.0)))
        phase_term = sum(dists) / max(1, len(dists))
        epi_term = sum(grad_epi) / max(1, len(grad_epi))
        dnfr = 0.6 * phase_term + 0.4 * epi_term
        _set_attr(d, ALIAS_DNFR, dnfr)


# -------------------------
# Selector glífico sesgado Si
# -------------------------

GLYPH_NAMES = (
    "A’L", "E’N", "I’L", "O’Z", "U’M", "R’A", "SH’A", "VA’L",
    "NU’L", "T’HOL", "Z’HIR", "NA’V", "RE’MESH",
)


def _glyph_dispatch(name: str):
    """Devuelve una función operador para el glifo si existe en operators.
    Permite alias sin diacríticos.
    """
    if OPS is None:
        return None
    # normalizamos nombre
    base = (
        name.replace("’", "'")
            .replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
            .replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
            .replace("Ñ", "N").replace("ñ", "n")
    )
    candidates = [name, base, base.replace("'", "")]  # tolerancia mínima
    for c in candidates:
        # convenciones comunes: op_AL, op_A_L, A_L, etc.
        for pref in ("op_", "", "apply_"):
            key = f"{pref}{c}".replace(" ", "").replace("-", "_").replace("/", "_").replace(".", "_")
            fn = getattr(OPS, key, None)
            if callable(fn):
                return fn
    return None


def default_glyph_selector(G, n, *, Si: float, dnfr: float, nu: float) -> str:
    """Selector canónico simple con sesgo por Si.
    Regla blanda:
      - Si alto (≥0.66)  → tender a I’L / R’A / U’M (estabilizar/propagar/acoplar)
      - Si medio (0.33..0.66) → NA’V / R’A / U’M (transición/adaptación)
      - Si bajo (<0.33) → O’Z / NA’V / Z’HIR (reorganizar)
    Se matiza con ΔNFR y νf para evitar decisiones triviales.
    """
    if Si >= 0.66:
        return "I’L" if dnfr < max(0.1, 0.25 * (1.0 + nu)) else "U’M"
    if Si >= 0.33:
        return "NA’V" if dnfr >= 0.2 else "R’A"
    # Si bajo
    return "O’Z" if dnfr < 0.75 else "Z’HIR"


# ---------------
# Paso de dinámica
# ---------------

def step(G, *, dt: float = 1.0, use_Si: bool = True, apply_glyphs: bool = True) -> None:
    """Ejecuta un paso de dinámica TNFR.
    Orden sugerido:
      1) Asegurar ΔNFR por nodo.
      2) (Mejora 1) compute_Si(G) si use_Si.
      3) Seleccionar glifo por nodo (callback o default sesgado por Si).
      4) Aplicar glifo (si apply_glyphs).
      5) Actualizar EPI por ecuación nodal.
    """
    if nx is None:
        raise RuntimeError("Se requiere networkx para dynamics.step")

    # 1) ΔNFR
    default_compute_delta_nfr(G)

    # 2) Si (Mejora 1)
    if use_Si:
        compute_Si(G, inplace=True)

    # 3) Selección glífica
    selector = G.graph.get("glyph_selector")
    if not callable(selector):
        selector = default_glyph_selector

    chosen: Dict[Any, str] = {}
    for n, d in G.nodes(data=True):
        Si = float(d.get("Si", 0.5))
        dnfr = _get_attr(d, ALIAS_DNFR, 0.0)
        nu = _get_attr(d, ALIAS_NU_F, 0.0)
        g = selector(G, n, Si=Si, dnfr=dnfr, nu=nu)
        if g not in GLYPH_NAMES:
            # Permitimos nombres alternativos a través del selector custom
            g = str(g)
        chosen[n] = g
        d["glyph"] = g  # trazabilidad

    # 4) Aplicación de glifos
    if apply_glyphs:
        for n, g in chosen.items():
            fn = _glyph_dispatch(g)
            if callable(fn):
                try:
                    fn(G, n)
                except Exception:
                    # no bloquea el paso si un glifo falla
                    pass

    # 5) Ecuación nodal
    update_epi_via_nodal_equation(G, dt=dt)

    # Hook opcional del usuario
    post = G.graph.get("after_step")
    if callable(post):
        try:
            post(G)
        except Exception:
            pass


# -----------------
# Bucle de simulación
# -----------------

def run(G, steps: int = 100, *, dt: float = 1.0, use_Si: bool = True, apply_glyphs: bool = True) -> None:
    """Ejecuta varios pasos de dinámica. Guarda un historial simple en G.graph.
    """
    G.graph.setdefault("history", {"C_steps": []})
    for k in range(steps):
        step(G, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs)
        # opcional: registrar un proxy de coherencia global C(t)
        try:
            # C(t) ↑ cuando |ΔNFR| y |dEPI_dt| ↓; métrica muy simple
            dnfr_mean = sum(abs(_get_attr(d, ALIAS_DNFR, 0.0)) for _, d in G.nodes(data=True)) / max(1, G.number_of_nodes())
            dEPI_mean = sum(abs(float(G.nodes[n].get("dEPI_dt", 0.0))) for n in G.nodes()) / max(1, G.number_of_nodes())
            C = 1.0 / (1.0 + dnfr_mean + dEPI_mean)
            G.graph["history"]["C_steps"].append(C)
        except Exception:
            pass
