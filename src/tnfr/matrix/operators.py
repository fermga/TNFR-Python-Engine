"""
operators.py — TNFR refactor (canónico, sin duplicados y con umbrales centralizados)

Notas canónicas y de higiene:
- θ es FASE estructural (sincronía). No es un umbral.
- Se eliminó el doble registro de REMESH en la rutina global.
- Se importan umbrales desde constants.py; nada de "magic numbers".
- Historiales de glifos se normalizan a STRINGS ("IL", "REMESH", ...),
  para compatibilidad con utilidades que comparan por texto.
- Se añadió `analizar_historial_glifico(...)` que unifica patrón + sintaxis.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple, Iterable

from constants import (
    EPI_MAX_GLOBAL,
    EPS_EPI_STABLE, EPS_DNFR_STABLE, EPS_DERIV_STABLE, EPS_ACCEL_STABLE,
    FRACTION_STABLE_REMESH,
    CODIGO_GLIFO, CODIGO_GLIFO_INV,
)


# -------------------------------
# Utilidades internas
# -------------------------------

def _safe_get_float(d: Dict[str, Any], k: str, default: float = 0.0) -> float:
    try:
        v = float(d.get(k, default))
        return v if v == v else default  # NaN check
    except Exception:
        return default


# -------------------------------
# Núcleo operativo de glifos
# -------------------------------

def aplicar_glifo(
    G: Any,
    nodo: Dict[str, Any],
    nodo_id: Any,
    nombre_glifo: str,
    historial_glifos_por_nodo: Dict[Any, List[Tuple[int, str]]],
    paso: int,
):
    """Aplica un glifo TNFR al nodo y registra historial.

    Nota canónica: θ es FASE estructural (sincronía), no umbral.
    """
    # Asegurar mínimos
    nodo.setdefault("EPI", 1.0)
    nodo.setdefault("Si", 0.5)
    nodo.setdefault("νf", 1.0)
    nodo.setdefault("ΔNFR", 0.0)
    nodo.setdefault("θ", 0.0)
    nodo.setdefault("estado", "activo")

    # Estado + preservación de θ_prev
    nodo["glifo"] = nombre_glifo
    nodo["estado"] = "silencio" if nombre_glifo == "SHA" else "activo"
    nodo["θ_prev"] = nodo.get("θ_prev", nodo.get("θ", 0.0))

    # Registro en historial por nodo (SIEMPRE como string)
    if historial_glifos_por_nodo is not None and paso is not None:
        historial_glifos_por_nodo.setdefault(nodo_id, []).append((paso, str(nombre_glifo)))

    # Registro local en el propio nodo (útil para EPIs compuestas)
    if paso is not None:
        nodo.setdefault("historial_glifos", []).append((paso, str(nombre_glifo)))

    # === Transformaciones estructurales por glifo TNFR ===
    if nombre_glifo == "AL":  # Emisión
        nodo["EPI"] += 0.2
        nodo["Si"] += 0.05
        nodo["νf"] *= 1.05
        nodo["ΔNFR"] *= 0.97

    elif nombre_glifo == "EN":  # Recepción
        nodo["Si"] += 0.08
        nodo["νf"] *= 0.95
        nodo["θ"] = max(0.0, nodo.get("θ", 0.0) - random.uniform(0.05, 0.15))

    elif nombre_glifo == "IL":  # Coherencia
        nodo["Si"] += 0.1
        nodo["EPI"] *= 1.05
        nodo["ΔNFR"] *= 0.95

    elif nombre_glifo == "OZ":  # Disonancia
        nodo["EPI"] *= 0.85
        nodo["ΔNFR"] *= 1.4
        nodo["νf"] *= 1.05
        nodo["Si"] *= 0.9

    elif nombre_glifo == "UM":  # Acoplamiento
        vecinos = list(G.neighbors(nodo_id)) if nodo_id in G else []
        if vecinos:
            media_vf = sum(_safe_get_float(G.nodes[v], "νf", 1.0) for v in vecinos) / len(vecinos)
            nodo["νf"] = (nodo["νf"] + media_vf) * 0.5
        nodo["ΔNFR"] *= 0.9

    elif nombre_glifo == "RA":  # Resonancia
        nodo["Si"] += 0.15
        nodo["EPI"] *= 1.05
        nodo["νf"] *= 1.02

    elif nombre_glifo == "SHA":  # Silencio
        nodo["estado"] = "silencio"
        nodo["νf"] *= 0.3
        nodo["ΔNFR"] *= 0.1
        nodo["Si"] *= 0.5
        nodo["EPI"] *= 0.9

    elif nombre_glifo == "VAL":  # Expansión
        nodo["EPI"] *= 1.15
        nodo["Si"] *= 1.08
        nodo["νf"] *= 1.05

    elif nombre_glifo == "NUL":  # Contracción
        nodo["EPI"] *= 0.82
        nodo["Si"] *= 0.92
        nodo["νf"] *= 0.92

    elif nombre_glifo == "THOL":  # Autoorganización
        nodo["νf"] *= 1.25
        nodo["Si"] *= 1.15
        nodo["θ"] = min(1.0, nodo.get("θ", 0.0) + random.uniform(0.1, 0.2))

    elif nombre_glifo == "ZHIR":  # Mutación
        nodo["EPI"] += 0.5
        nodo["νf"] *= 1.2
        nodo["θ"] = min(1.0, nodo.get("θ", 0.0) + random.uniform(0.15, 0.3))
        nodo["Si"] *= 1.1

    elif nombre_glifo == "NAV":  # Nacimiento / transición
        nodo["νf"] *= 1.08
        nodo["ΔNFR"] *= 0.9
        nodo["Si"] += 0.1
        if nodo.get("estado") == "silencio":
            nodo["estado"] = "activo"

    elif nombre_glifo == "REMESH":  # Recursividad / memoria
        nodo["EPI"] = (
            _safe_get_float(nodo, "EPI_prev", nodo["EPI"]) + _safe_get_float(nodo, "EPI_prev2", nodo["EPI"])
        ) / 2.0
        nodo["Si"] *= 0.98
        nodo["νf"] *= 0.98

    # Clamps suaves post-transformación
    nodo["EPI"] = min(max(nodo["EPI"], 0.0), EPI_MAX_GLOBAL)
    nodo["θ"] = max(0.0, min(1.0, nodo.get("θ", 0.0)))


# -------------------------------
# Análisis glífico
# -------------------------------

def evaluar_patron_glifico(glifos: List[str]) -> Dict[str, Any]:
    glifos = [str(g).upper() for g in glifos]
    patron = " → ".join(glifos)

    analisis = {
        "ciclos_REMESH": glifos.count("REMESH"),
        "uso_THOL": glifos.count("THOL"),
        "uso_ZHIR": glifos.count("ZHIR"),
        "latencia_prolongada": any(
            glifos[i] == "SHA" and glifos[i + 1] == "SHA" for i in range(len(glifos) - 1)
        ),
        "inicio_creativo": bool(glifos) and glifos[0] == "AL",
        "coherencia_expansiva": ("IL" in glifos and "VAL" in glifos),
        "disonancia_sostenida": any(
            glifos[i] == "OZ" and glifos[i + 1] == "OZ" for i in range(len(glifos) - 1)
        ),
        "patron_glifico": patron,
        "tipo_nodal": (
            "creador" if glifos and glifos[0] == "AL" else
            "mutante" if "ZHIR" in glifos else
            "recursivo" if glifos.count("REMESH") > 2 else
            "expansivo" if "VAL" in glifos else
            "latente"
        ),
    }
    return analisis


def interpretar_sintaxis_glifica(historial: Dict[Any, List[Tuple[int, str]]]) -> Dict[Any, Dict[str, Any]]:
    sintaxis: Dict[Any, Dict[str, Any]] = {}
    for nodo, secuencia in historial.items():
        trayecto = [str(g).upper() for _, g in secuencia]
        if not trayecto:
            continue
        transiciones = list(zip(trayecto, trayecto[1:]))
        ciclos_val_nul = sum(
            1 for i in range(len(trayecto) - 2)
            if trayecto[i] == "VAL" and trayecto[i + 1] == "NUL" and trayecto[i + 2] == "VAL"
        )

        tipo = "desconocido"
        if "ZHIR" in trayecto:
            tipo = "mutante"
        elif "REMESH" in trayecto:
            tipo = "recursivo"
        elif ciclos_val_nul >= 2:
            tipo = "pulsante"
        elif trayecto.count("IL") > 2:
            tipo = "estabilizador"

        sintaxis[nodo] = {
            "trayectoria": trayecto,
            "transiciones": transiciones,
            "mutaciones": trayecto.count("ZHIR"),
            "colapsos": trayecto.count("SHA"),
            "ciclos_val_nul": ciclos_val_nul,
            "diversidad_glifica": len(set(trayecto)),
            "tipo_nodal": tipo,
        }
    return sintaxis


def analizar_historial_glifico(historial: Dict[Any, List[Tuple[int, str]]]) -> Dict[Any, Dict[str, Any]]:
    """Unifica análisis de patrón + sintaxis glífica por nodo."""
    sint = interpretar_sintaxis_glifica(historial)
    resumen: Dict[Any, Dict[str, Any]] = {}
    for nodo_id, datos in sint.items():
        trayecto = datos.get("trayectoria", [])
        patron = evaluar_patron_glifico(trayecto)
        resumen[nodo_id] = {
            **datos,
            "analitica_patron": patron,
            "patron_glifico": patron.get("patron_glifico", ""),
        }
    return resumen


# -------------------------------
# Selección de glifo por estructura
# -------------------------------

def glifo_por_estructura(nodo: Dict[str, Any], G: Any) -> str | None:
    n_id = nodo.get("nodo", None)
    vecinos = list(G.neighbors(n_id)) if (n_id is not None and n_id in G) else []

    # 1. SHA – Silencio ante alta disonancia
    if _safe_get_float(nodo, "EPI", 1.0) < 0.5 and abs(_safe_get_float(nodo, "ΔNFR", 0.0)) > 0.8:
        return "SHA"

    # 2. NAV – Activación desde silencio
    if nodo.get("estado") == "silencio" and abs(_safe_get_float(nodo, "ΔNFR", 0.0) - _safe_get_float(nodo, "νf", 1.0)) < 0.05:
        return "NAV"

    # 3. AL – Emisión si es latente y sensible
    if nodo.get("estado") == "latente" and _safe_get_float(nodo, "Si", 0.0) < 0.2 and _safe_get_float(nodo, "νf", 1.0) > 1.0:
        return "AL"

    # 4. EN – Recepción ante apertura sensible
    if _safe_get_float(nodo, "ΔNFR", 0.0) > 0.6 and _safe_get_float(nodo, "EPI", 1.0) > 1.0 and _safe_get_float(nodo, "Si", 0.0) < 0.3:
        return "EN"

    # 5. OZ – Disonancia fuerte
    if abs(_safe_get_float(nodo, "ΔNFR", 0.0)) > 1.0 and _safe_get_float(nodo, "EPI", 1.0) > 1.0:
        return "OZ"

    # 6. ZHIR – Mutación por cambio abrupto
    if abs(_safe_get_float(nodo, "EPI", 1.0) - _safe_get_float(nodo, "EPI_prev", nodo.get("EPI", 1.0))) > 0.5 and _safe_get_float(nodo, "Si", 0.0) > 0.5:
        return "ZHIR"

    # 7. VAL – Expansión estructural
    if _safe_get_float(nodo, "Si", 0.0) > 0.6 and _safe_get_float(nodo, "EPI", 1.0) > 1.2:
        return "VAL"

    # 8. NUL – Contracción por exceso
    if _safe_get_float(nodo, "EPI", 1.0) > 1.3 and _safe_get_float(nodo, "Si", 0.0) < 0.4:
        return "NUL"

    # 9. THOL – Autoorganización
    if abs(_safe_get_float(nodo, "EPI", 1.0) - _safe_get_float(nodo, "EPI_prev2", nodo.get("EPI", 1.0))) > 0.2 and abs(_safe_get_float(nodo, "ΔNFR", 0.0)) < 0.1:
        return "THOL"

    # 10. IL – Coherencia estable
    if abs(_safe_get_float(nodo, "ΔNFR", 0.0)) < 0.05 and abs(_safe_get_float(nodo, "EPI", 1.0) - _safe_get_float(nodo, "EPI_prev", nodo.get("EPI", 1.0))) < 0.05:
        return "IL"

    # 11. RA – Resonancia coherente
    if str(nodo.get("glifo", "")).upper() == "IL" and _safe_get_float(nodo, "Si", 0.0) > 0.5 and _safe_get_float(nodo, "νf", 1.0) > 1.2:
        return "RA"

    # 12. UM – Acoplamiento con vecinos
    for v in vecinos:
        if abs(_safe_get_float(nodo, "νf", 1.0) - _safe_get_float(G.nodes[v], "νf", 1.0)) < 0.05:
            return "UM"

    # 13. REMESH – Recursividad (si ya hay historial)
    hist = nodo.get("historial_glifos", [])
    if (
        len(hist) >= 3
        and str(hist[-1][1]).upper() == str(hist[-2][1]).upper() == str(hist[-3][1]).upper()
        and abs(_safe_get_float(nodo, "EPI", 1.0) - _safe_get_float(nodo, "EPI_prev", nodo.get("EPI", 1.0))) < 0.05
    ):
        return "REMESH"

    return None  # si no se detecta un glifo resonante


# -------------------------------
# Transiciones canónicas
# -------------------------------

def transicion_glifica_canonica(nodo: Dict[str, Any]) -> str | None:
    glifo = str(nodo.get("glifo", "")).upper()

    if glifo == "ZHIR":
        if _safe_get_float(nodo, "νf", 1.0) > 1.5 and _safe_get_float(nodo, "EPI", 1.0) > 0.7 * EPI_MAX_GLOBAL:
            return "VAL"
        elif _safe_get_float(nodo, "ΔNFR", 0.0) < 0:
            return "RA"
        else:
            return "NAV"

    elif glifo == "IL":
        if _safe_get_float(nodo, "νf", 1.0) > 1.2 and _safe_get_float(nodo, "Si", 0.0) > 0.4:
            return "RA"

    elif glifo == "OZ":
        if _safe_get_float(nodo, "EPI", 1.0) > 0.63 * EPI_MAX_GLOBAL and _safe_get_float(nodo, "Si", 0.0) > 0.3:
            return "THOL"

    elif glifo == "NAV":
        if abs(_safe_get_float(nodo, "ΔNFR", 0.0)) < 0.1:
            return "IL"

    elif glifo == "RA":
        if _safe_get_float(nodo, "Si", 0.0) > 0.6 and _safe_get_float(nodo, "EPI", 1.0) > 0.75 * EPI_MAX_GLOBAL:
            return "REMESH"

    elif glifo == "VAL":
        if _safe_get_float(nodo, "EPI", 1.0) > 0.86 * EPI_MAX_GLOBAL and _safe_get_float(nodo, "Si", 0.0) > 0.4:
            return "NUL"

    elif glifo == "AL":
        if _safe_get_float(nodo, "Si", 0.0) > 0.3 and _safe_get_float(nodo, "ΔNFR", 0.0) < 0.2:
            return "UM"

    return None


# -------------------------------
# Acoplamiento y EPIs compuestas
# -------------------------------

def acoplar_nodos(G: Any) -> None:
    for n in list(G.nodes):
        vecinos = list(G.neighbors(n))
        if not vecinos:
            vecinos = [v for v in G.nodes if v != n]
        Si_vecinos = [_safe_get_float(G.nodes[v], "Si", 0.0) for v in vecinos]
        if Si_vecinos:
            G.nodes[n]["Si"] = (sum(Si_vecinos) / len(Si_vecinos)) * 0.9 + _safe_get_float(G.nodes[n], "Si", 0.0) * 0.1
        for v in vecinos:
            if abs(_safe_get_float(G.nodes[n], "θ", 0.0) - _safe_get_float(G.nodes[v], "θ", 0.0)) < 0.1:
                G.nodes[n]["ΔNFR"] *= 0.95


def detectar_EPIs_compuestas(G: Any, umbrales: Dict[str, float] | None = None) -> List[Dict[str, Any]]:
    # Si no se pasan umbrales, usar valores por defecto
    umbral_theta = (umbrales or {}).get("θ_conexion", 0.12)
    umbral_si = (umbrales or {}).get("Si_conexion", 0.2)

    compuestas: List[Dict[str, Any]] = []
    nodos_por_glifo_y_paso: Dict[Tuple[int, str], List[Any]] = {}

    for n in G.nodes:
        historial = G.nodes[n].get("historial_glifos", [])
        for paso, glifo in historial:
            clave = (int(paso), str(glifo))
            nodos_por_glifo_y_paso.setdefault(clave, []).append(n)

    for (paso, glifo), nodos_en_glifo in nodos_por_glifo_y_paso.items():
        if len(nodos_en_glifo) < 3:
            continue

        grupo_coherente: List[Any] = []
        for i, ni in enumerate(nodos_en_glifo):
            for nj in nodos_en_glifo[i + 1 :]:
                θi, θj = _safe_get_float(G.nodes[ni], "θ", 0.0), _safe_get_float(G.nodes[nj], "θ", 0.0)
                Sii, Sij = _safe_get_float(G.nodes[ni], "Si", 0.0), _safe_get_float(G.nodes[nj], "Si", 0.0)
                if abs(θi - θj) < umbral_theta and abs(Sii - Sij) < umbral_si:
                    grupo_coherente.extend([ni, nj])

        grupo_final = list(set(grupo_coherente))
        if len(grupo_final) >= 3:
            compuestas.append({
                "paso": paso,
                "glifo": glifo,
                "nodos": grupo_final,
                "tipo": clasificar_epi(glifo),
            })

    return compuestas


def clasificar_epi(glifo: str) -> str:
    glifo = str(glifo).upper()
    if glifo in ["IL", "RA", "REMESH"]:
        return "coherente"
    elif glifo in ["ZHIR", "VAL", "NUL"]:
        return "mutante"
    elif glifo in ["SHA", "OZ"]:
        return "disonante"
    else:
        return "otro"


# -------------------------------
# REMESH global (sin duplicar historial)
# -------------------------------

def aplicar_remesh_red(G: Any, historial_glifos_por_nodo: Dict[Any, List[Tuple[int, str]]], paso: int) -> None:
    for n in list(G.nodes):
        aplicar_glifo(G, G.nodes[n], n, "REMESH", historial_glifos_por_nodo, paso)


def aplicar_remesh_si_estabilizacion_global(
    G: Any,
    historial_glifos_por_nodo: Dict[Any, List[Tuple[int, str]]],
    historia_glifos: List[str],
    paso: int,
) -> None:
    if len(G) == 0:
        return

    nodos_estables = 0
    for n in G.nodes:
        nodo = G.nodes[n]
        estabilidad_epi = abs(_safe_get_float(nodo, "EPI", 0.0) - _safe_get_float(nodo, "EPI_prev", 0.0)) < EPS_EPI_STABLE
        estabilidad_nfr = abs(_safe_get_float(nodo, "ΔNFR", 0.0)) < EPS_DNFR_STABLE
        estabilidad_dEPI = abs(_safe_get_float(nodo, "dEPI_dt", 0.0)) < EPS_DERIV_STABLE
        estabilidad_acel = abs(_safe_get_float(nodo, "d2EPI_dt2", 0.0)) < EPS_ACCEL_STABLE
        if all([estabilidad_epi, estabilidad_nfr, estabilidad_dEPI, estabilidad_acel]):
            nodos_estables += 1

    fraccion_estables = nodos_estables / len(G)
    if fraccion_estables > FRACTION_STABLE_REMESH:
        aplicar_remesh_red(G, historial_glifos_por_nodo, paso)
        # IMPORTANTE: No duplicar historial aquí. Si deseas logging global, usa SOLO historia_glifos.
        # (Descomenta la siguiente línea si quieres un log global textual del REMESH)
        # historia_glifos.extend([f"{paso},{n},REMESH" for n in G.nodes])


# -------------------------------
# Normalización de historiales glíficos
# -------------------------------

def limpiar_glifo(glifo_raw: Any) -> str:
    """Limpia glifos que pueden tener comillas o separadores extra (A'L → AL, etc.)."""
    s = str(glifo_raw).strip().strip("'").strip('"')
    correcciones = {
        "RE'MESH": "REMESH",
        "T'HOL": "THOL",
        "Z'HIR": "ZHIR",
        "A'L": "AL",
        "E'N": "EN",
        "I'L": "IL",
        "O'Z": "OZ",
        "U'M": "UM",
        "R'A": "RA",
        "SH'A": "SHA",
        "VA'L": "VAL",
        "NU'L": "NUL",
        "NA'V": "NAV",
    }
    for k, v in correcciones.items():
        if s == k or s.replace("'", "") == v:
            return v
    return s.upper()


def normalizar_historial_glifos(
    historial_glifos_por_nodo: Dict[Any, List[Tuple[Any, Any]]],
    analizar_dinamica: bool = False,
    expandido: bool = False,
):
    """Normaliza historiales a forma: List[(paso_int, GLIFO_STR)].

    - Acepta entradas (paso, "GLIFO"), ("GLIFO", paso) o (paso, codigo_int).
    - Convierte códigos con CODIGO_GLIFO_INV.
    - Opcionalmente devuelve análisis dinámico y/o historial expandido.
    """
    resumen_dinamico: Dict[Any, Dict[str, Any]] = {}
    historial_expandido: Dict[Any, List[Dict[str, Any]]] = {}

    for nodo_id, historial in list(historial_glifos_por_nodo.items()):
        nuevo_historial: List[Tuple[int, str]] = []
        historial_completo: List[Dict[str, Any]] = []
        glifos_validos: List[str] = []

        for entrada in list(historial):
            if not isinstance(entrada, (list, tuple)) or len(entrada) != 2:
                continue
            a, b = entrada
            paso: int | None = None
            glifo_str: str | None = None

            # (paso, "GLIFO")
            if isinstance(a, (int, float)) and isinstance(b, str):
                paso = int(a)
                glifo_str = limpiar_glifo(b)

            # ("GLIFO", paso)
            elif isinstance(a, str) and isinstance(b, (int, float)):
                paso = int(b)
                glifo_str = limpiar_glifo(a)

            # (paso, codigo_int) o (codigo_int, paso)
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if int(b) in CODIGO_GLIFO_INV:
                    paso = int(a)
                    glifo_str = CODIGO_GLIFO_INV[int(b)]
                elif int(a) in CODIGO_GLIFO_INV:
                    paso = int(b)
                    glifo_str = CODIGO_GLIFO_INV[int(a)]

            if paso is None or glifo_str is None:
                continue

            nuevo_historial.append((paso, glifo_str))
            historial_completo.append({"paso": paso, "glifo": glifo_str, "codigo": CODIGO_GLIFO.get(glifo_str)})
            glifos_validos.append(glifo_str)

        # Ordenar y asignar
        nuevo_historial.sort(key=lambda x: x[0])
        historial_glifos_por_nodo[nodo_id] = nuevo_historial
        historial_expandido[nodo_id] = historial_completo

        if analizar_dinamica and glifos_validos:
            resumen_dinamico[nodo_id] = evaluar_patron_glifico(glifos_validos)

    if analizar_dinamica and expandido:
        return resumen_dinamico, historial_expandido
    if expandido:
        return historial_expandido
    if analizar_dinamica:
        return resumen_dinamico
    return historial_glifos_por_nodo


__all__ = [
    "aplicar_glifo",
    "evaluar_patron_glifico",
    "glifo_por_estructura",
    "transicion_glifica_canonica",
    "acoplar_nodos",
    "detectar_EPIs_compuestas",
    "clasificar_epi",
    "normalizar_historial_glifos",
    "interpretar_sintaxis_glifica",
    "aplicar_remesh_red",
    "aplicar_remesh_si_estabilizacion_global",
    "limpiar_glifo",
    "analizar_historial_glifico",
]
