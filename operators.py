"""
operators.py — TNFR canónica

Glifos canónicos (13) implementados como operadores nodales sobre el grafo G.
Todos aceptan la firma (G, n, **kw) y mutan atributos del nodo n.

Alineación con Mejora 1:
- No calculan Si (lo hace helpers.compute_Si), pero **no rompen** su semántica.
- Procuran que I’L tienda a reducir ΔNFR, O’Z a incrementarlo, U’M a sincronizar
  fase, R’A a propagar EPI, etc., según las definiciones TNFR.

Notas:
- Se usan alias canónicos para atributos: EPI/PSI, νf, θ, ΔNFR.
- Se añade un pequeño helper para historiales: EPI_hist y θ_hist (opcional).
- Las operaciones son locales; donde aplica, usan vecinos inmediatos.
- Todas las modificaciones son suaves (factores 0..1) para evitar inestabilidad.

Autor: TNFR | Teoría de la naturaleza fractal resonante
"""
from __future__ import annotations

from typing import Any, Dict, Iterable
import math

# -----------------------------
# Alias canónicos de atributos
# -----------------------------
ALIAS_EPI = ("EPI", "PSI", "psi", "epi")
ALIAS_NU_F = ("νf", "nu_f", "nu-f", "nu", "freq", "frequency")
ALIAS_THETA = ("θ", "theta", "fase", "phi", "phase")
ALIAS_DNFR = ("ΔNFR", "delta_nfr", "dnfr", "gradiente", "grad")


# -----------------------------
# Utilidades generales
# -----------------------------

def _get_attr(d: Dict[str, Any], aliases: Iterable[str], default: float = 0.0) -> float:
    for k in aliases:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)


def _set_attr(d: Dict[str, Any], aliases: Iterable[str], value: float) -> None:
    key = next(iter(aliases))
    d[key] = float(value)


def _append_hist(d: Dict[str, Any], key: str, value: float) -> None:
    try:
        hist = d.get(key)
        if not isinstance(hist, list):
            d[key] = [value]
        else:
            hist.append(float(value))
    except Exception:
        d[key] = [float(value)]


def _wrap_angle(x: float) -> float:
    two_pi = 2.0 * math.pi
    ax = float(x)
    if abs(ax) > two_pi * 3.0:
        ax = math.radians(ax)
    return ax % (2.0 * math.pi)


def _phase_dist01(a: float, b: float) -> float:
    i, j = _wrap_angle(a), _wrap_angle(b)
    d = abs(i - j)
    d = min(d, 2.0 * math.pi - d)
    return d / math.pi  # 0..1


def _neighbors(G, n):
    if G.is_directed():
        return list(G.predecessors(n)) + list(G.successors(n))
    return list(G.neighbors(n))


def _mean(vals):
    vals = list(vals)
    return sum(vals) / max(len(vals), 1)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ------------------------------------------------
# Glifos canónicos (funciones op_*)
# ------------------------------------------------

# A’L — Emisión: kick inicial estructurante sobre EPI en función de νf y θ.
def op_AL(G, n, **kw):
    d = G.nodes[n]
    nu = _get_attr(d, ALIAS_NU_F, 0.0)
    th = _get_attr(d, ALIAS_THETA, 0.0)
    epi = _get_attr(d, ALIAS_EPI, 0.0)
    kick = 0.25 * nu * math.cos(_wrap_angle(th))
    _set_attr(d, ALIAS_EPI, epi + kick)
    _append_hist(d, "EPI_hist", epi)


# E’N — Recepción: aproxima EPI al entorno (media vecinal ponderada por fase).
def op_EN(G, n, **kw):
    d = G.nodes[n]
    epi_i = _get_attr(d, ALIAS_EPI, 0.0)
    th_i = _get_attr(d, ALIAS_THETA, 0.0)
    neighs = _neighbors(G, n)
    if not neighs:
        return
    weights = []
    epis = []
    for m in neighs:
        dm = G.nodes[m]
        th_m = _get_attr(dm, ALIAS_THETA, 0.0)
        w = 1.0 - _phase_dist01(th_i, th_m)  # más peso si están en fase
        weights.append(w)
        epis.append(_get_attr(dm, ALIAS_EPI, 0.0))
    W = sum(weights) or 1.0
    target = sum(e * w for e, w in zip(epis, weights)) / W
    epi_next = epi_i + 0.35 * (target - epi_i)
    _set_attr(d, ALIAS_EPI, epi_next)


# I’L — Coherencia: reduce ΔNFR y amortigua derivadas (estabiliza).
def op_IL(G, n, **kw):
    d = G.nodes[n]
    dnfr = _get_attr(d, ALIAS_DNFR, 0.0)
    d[ALIAS_DNFR[0]] = 0.7 * dnfr  # reduce necesidad de reorganización
    # si hay derivada previa, amortiguamos
    if "dEPI_dt" in d:
        d["dEPI_dt"] = 0.6 * float(d["dEPI_dt"])


# O’Z — Disonancia: eleva ΔNFR (tensión reorganizadora).
def op_OZ(G, n, **kw):
    d = G.nodes[n]
    dnfr = _get_attr(d, ALIAS_DNFR, 0.0)
    nu = _get_attr(d, ALIAS_NU_F, 0.0)
    # incrementamos dnfr y, si queda por debajo de νf, lo llevamos cerca.
    dnfr_next = dnfr * 1.25 + 0.1
    if dnfr_next < 0.8 * max(nu, 1e-6):
        dnfr_next = 0.8 * max(nu, 1e-6)
    d[ALIAS_DNFR[0]] = dnfr_next


# U’M — Acoplamiento: sincroniza θ con vecinos, pequeño corrimiento hacia media.
def op_UM(G, n, **kw):
    d = G.nodes[n]
    th_i = _get_attr(d, ALIAS_THETA, 0.0)
    neighs = _neighbors(G, n)
    if not neighs:
        return
    # Promedio circular de fases
    xs, ys = 0.0, 0.0
    for m in neighs:
        th_m = _wrap_angle(_get_attr(G.nodes[m], ALIAS_THETA, 0.0))
        xs += math.cos(th_m)
        ys += math.sin(th_m)
    mean_ang = math.atan2(ys, xs) if xs or ys else th_i
    # desplazamiento suave
    th_next = _wrap_angle(th_i + 0.35 * (_wrap_angle(mean_ang - th_i)))
    _set_attr(d, ALIAS_THETA, th_next)
    _append_hist(d, "θ_hist", th_i)


# R’A — Resonancia: propaga coherencia (EPI) si hay buena sincronía de fase.
def op_RA(G, n, **kw):
    d = G.nodes[n]
    epi_i = _get_attr(d, ALIAS_EPI, 0.0)
    th_i = _get_attr(d, ALIAS_THETA, 0.0)
    neighs = _neighbors(G, n)
    if not neighs:
        return
    syncs = [1.0 - _phase_dist01(th_i, _get_attr(G.nodes[m], ALIAS_THETA, 0.0)) for m in neighs]
    gain = _mean(syncs)  # 0..1
    # R’A NO toca θ; sólo refuerza EPI en proporción a la sincronía
    epi_next = epi_i + 0.2 * gain * abs(epi_i)
    _set_attr(d, ALIAS_EPI, epi_next)


# SH’A — Silencio: repliegue; baja νf y apaga derivadas sin borrar forma.
def op_SHA(G, n, **kw):
    d = G.nodes[n]
    nu = _get_attr(d, ALIAS_NU_F, 0.0)
    nu_next = 0.5 * nu
    _set_attr(d, ALIAS_NU_F, nu_next)
    if "dEPI_dt" in d:
        d["dEPI_dt"] = 0.25 * float(d["dEPI_dt"])


# VA’L — Expansión: amplifica EPI (crecimiento multiescalar).
def op_VAL(G, n, **kw):
    d = G.nodes[n]
    epi = _get_attr(d, ALIAS_EPI, 0.0)
    _set_attr(d, ALIAS_EPI, epi * 1.15)


# NU’L — Contracción: densifica; reduce soporte (si existe) y comprime EPI.
def op_NUL(G, n, **kw):
    d = G.nodes[n]
    epi = _get_attr(d, ALIAS_EPI, 0.0)
    _set_attr(d, ALIAS_EPI, epi * 0.92)
    # si hubiera atributo de soporte, lo reducimos suavemente
    supp = d.get("Supp", None)
    if isinstance(supp, (int, float)):
        d["Supp"] = float(supp) * 0.95


# T’HOL — Autoorganización: bifurca hacia estructura estable (reduce aceleración).
def op_THOL(G, n, **kw):
    d = G.nodes[n]
    # amortiguamos segunda derivada si estuviera medida vía dEPI_dt previo/posterior
    if "dEPI_dt" in d and "EPI_prev" in d:
        d["dEPI_dt"] = 0.5 * float(d["dEPI_dt"])
    # pequeño centrado de fase en atractores discretos (0, ±π/2, π)
    th = _wrap_angle(_get_attr(d, ALIAS_THETA, 0.0))
    attractors = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]
    nearest = min(attractors, key=lambda a: abs(a - th))
    th_next = th + 0.3 * (nearest - th)
    _set_attr(d, ALIAS_THETA, _wrap_angle(th_next))


# Z’HIR — Mutación: cambio de fase no destructivo; rota θ y recalibra νf.
def op_ZHIR(G, n, **kw):
    d = G.nodes[n]
    th = _wrap_angle(_get_attr(d, ALIAS_THETA, 0.0))
    dth = 0.5 * math.pi  # rotación de 90°
    _set_attr(d, ALIAS_THETA, _wrap_angle(th + dth))
    # recalibración suave de νf para sostener la nueva fase
    nu = _get_attr(d, ALIAS_NU_F, 0.0)
    _set_attr(d, ALIAS_NU_F, 0.9 * nu + 0.1 * (nu + 0.25))


# NA’V — Transición: lleva ΔNFR hacia νf (inestabilidad creativa controlada).
def op_NAV(G, n, **kw):
    d = G.nodes[n]
    nu = _get_attr(d, ALIAS_NU_F, 0.0)
    dnfr = _get_attr(d, ALIAS_DNFR, 0.0)
    target = max(nu, 1e-6)
    d[ALIAS_DNFR[0]] = dnfr + 0.4 * (target - dnfr)


# RE’MESH — Recursividad/memoria: reinyecta estado con retardo τ si existe.
def op_REMESH(G, n, **kw):
    d = G.nodes[n]
    tau = int(G.graph.get("tau", 1))
    if tau <= 0:
        return
    hist = d.get("EPI_hist")
    if isinstance(hist, list) and len(hist) > tau:
        epi_tau = float(hist[-tau])
        epi = _get_attr(d, ALIAS_EPI, 0.0)
        _set_attr(d, ALIAS_EPI, epi + 0.5 * (epi_tau - epi))


# ------------------------------------------------
# Alias alternativos (por compatibilidad con despachador)
# ------------------------------------------------
# El despachador en dynamics prueba: op_<NAME>, <NAME>, apply_<NAME>,
# con NAME normalizado (p. ej., I’L → IL). Definimos alias sencillos.
apply_AL = op_AL
apply_EN = op_EN
apply_IL = op_IL
apply_OZ = op_OZ
apply_UM = op_UM
apply_RA = op_RA
apply_SHA = op_SHA
apply_VAL = op_VAL
apply_NUL = op_NUL
apply_THOL = op_THOL
apply_ZHIR = op_ZHIR
apply_NAV = op_NAV
apply_REMESH = op_REMESH
