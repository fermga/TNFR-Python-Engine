"""
helpers.py — TNFR refactor (canónico y sin redundancias)

Este módulo implementa utilidades operativas desde la Teoría de la Naturaleza Fractal Resonante (TNFR).
Notas canónicas:
- θ es FASE estructural (sincronía relativa), no un umbral.
- La emergencia de forma se evalúa por estabilidad de EPI y ΔNFR (ecuación nodal).
- Historiales de glifos deben almacenarse SIEMPRE como strings ("AL", "REMESH", etc.).

Cambios clave del refactor:
- Docstrings y tipados suaves para mayor claridad.
- Umbrales centralizados con import opcional desde constants.py (con fallbacks seguros).
- Comparaciones de glifos robustas (normalización a string).
- Limpieza de duplicados en historial_macronodos.
- Ligeras robusteces numéricas (clamps y defaults). 
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Iterable, Any

from operators import aplicar_glifo
from constants import glifo_categoria
import numpy as np

# --- Umbrales/constantes (intento de import desde constants; si no existen, usar fallbacks canónicos) ---
try:  # estos pueden existir si ya aplicaste el refactor en constants.py
    from constants import (
        EPS_EPI_STABLE,
        EPS_DNFR_STABLE,
        EPS_DERIV_STABLE,
        EPS_ACCEL_STABLE,
        FRACTION_STABLE_REMESH,
    )
except Exception:  # fallbacks por compatibilidad hacia atrás
    EPS_EPI_STABLE = 0.01
    EPS_DNFR_STABLE = 0.05
    EPS_DERIV_STABLE = 0.01
    EPS_ACCEL_STABLE = 0.01
    FRACTION_STABLE_REMESH = 0.8

# Tolerancias locales específicas de este módulo
THETA_MUTATION_TOL = 0.2   # salto de fase para Z'HIR
SI_RA_THRESHOLD    = 0.8   # Si alto + UM previo => R'A
PHASE_GROUP_TOL    = 0.1   # acople de fase para agrupar macronodos
EPI_GROUP_TOL_HARD = 0.2   # cercanía de EPI para agrupar macronodos
EMERGENCIA_VF_FRAC = 0.9   # vf > media_vf*0.9 indica nodo alto en su rango


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def emergencia_nodal(nodo: Dict[str, Any], media_vf: float, std_dNFR: float) -> bool:
    """Evalúa si un nodo está en emergencia operativa (alto νf y bajo |ΔNFR|).

    Criterio práctico TNFR: si su νf supera un porcentaje de la media y su ΔNFR
    está por debajo de la desviación típica del entorno.
    """
    vf = _as_float(nodo.get("νf", 1.0), 1.0)
    dNFR = _as_float(nodo.get("ΔNFR", 0.0), 0.0)

    mvf = _as_float(media_vf, vf)
    std = abs(_as_float(std_dNFR, 1.0)) or 1.0

    return (vf > mvf * EMERGENCIA_VF_FRAC) and (abs(dNFR) < std)


def promover_emergente(
    nodo_id: Any,
    G: Any,
    paso: int,
    historial_glifos_por_nodo: Dict[Any, List[Tuple[int, str]]],
    historia_glifos: List[str],
) -> None:
    """Decide y aplica el glifo emergente canónico para un nodo.

    Decisión glífica (TNFR operativa):
    - REMESH si EPI está estable y |ΔNFR| es bajo (memoria estructural activa).
    - Z'HIR si hay salto de fase θ significativo (cambio de identidad).
    - R'A si Si es alto y venimos de U'M (propagación coherente en red).
    - T'HOL en caso contrario (autoorganización local/pasaje intermedio).

    Nota: θ es fase estructural (sincronía). Se conservan valores previos para
detectar cambios significativos.
    """
    if nodo_id not in G:
        return

    nodo = G.nodes[nodo_id]

    # Asegurar valores previos
    if "EPI_prev" not in nodo:
        epi_actual = nodo.get("EPI", 1.0)
        nodo["EPI_prev"] = epi_actual if np.isfinite(epi_actual) else 1.0
    if "θ_prev" not in nodo:
        nodo["θ_prev"] = nodo.get("θ", 0.0)

    # Lecturas seguras
    EPI     = _as_float(nodo.get("EPI", 1.0), 1.0)
    EPI_prev= _as_float(nodo.get("EPI_prev", EPI), EPI)
    dNFR    = _as_float(nodo.get("ΔNFR", 0.0), 0.0)
    theta   = _as_float(nodo.get("θ", 0.0), 0.0)
    theta_p = _as_float(nodo.get("θ_prev", theta), theta)
    Si      = _as_float(nodo.get("Si", 0.5), 0.5)

    # Decisión canónica
    if abs(EPI - EPI_prev) < EPS_EPI_STABLE and abs(dNFR) < EPS_DNFR_STABLE:
        glifo = "REMESH"
    elif abs(theta - theta_p) > THETA_MUTATION_TOL:
        glifo = "ZHIR"
    elif Si > SI_RA_THRESHOLD and str(nodo.get("glifo", "")).upper() == "UM":
        glifo = "RA"
    else:
        glifo = "THOL"

    aplicar_glifo(G, nodo, nodo_id, glifo, historial_glifos_por_nodo, paso)
    historia_glifos.append(f"{paso},{nodo_id},{glifo}")
    nodo["glifo"] = glifo
    nodo["categoria"] = glifo_categoria.get(glifo, "ninguna")


def detectar_nodos_pulsantes(
    historial_glifos_por_nodo: Dict[Any, List[Tuple[int, str]]],
    min_ciclos: int = 3,
) -> List[Any]:
    """Identifica nodos con ciclos VAL→NUL repetidos (oscilación pulsante)."""
    nodos_maestros: List[Any] = []
    for nodo_id, eventos in historial_glifos_por_nodo.items():
        glifos = [str(g).upper() for _, g in eventos]
        ciclos = 0
        for i in range(len(glifos) - 1):
            if glifos[i] == "VAL" and glifos[i + 1] == "NUL":
                ciclos += 1
        if ciclos >= min_ciclos:
            nodos_maestros.append(nodo_id)
    return nodos_maestros


def detectar_macronodos(
    G: Any,
    historial_glifos_por_nodo: Dict[Any, List[Tuple[int, str]]],
    epi_compuestas: Iterable[Dict[str, Any]],
    paso: int,
    umbral_coherencia: float = 0.05,
    visualizar: bool = True,
):
    """Agrupa nodos con trayectorias glíficas y estado estructural similares para formar macronodos.

    Criterios de agrupación (pragmáticos y canónicos):
    - Misma secuencia de glifos en la ventana reciente.
    - Fase θ cercana (< PHASE_GROUP_TOL) y EPI cercana (|ΔEPI| < umbral_coherencia).
    """
    historial_macronodos: List[Dict[str, Any]] = []
    candidatos: List[Tuple[Any, List[str]]] = []

    # Preselección por patrón glífico reciente
    for n in list(G.nodes):
        historial = historial_glifos_por_nodo.get(n, [])
        if len(historial) >= 5:
            glifos_ultimos = [str(g) for _, g in historial[-5:]]
            candidatos.append((n, glifos_ultimos))

    grupos: List[List[Any]] = []
    visitados: set = set()
    for n1, glifos1 in candidatos:
        if n1 in visitados:
            continue
        grupo = [n1]
        for n2, glifos2 in candidatos:
            if n1 == n2 or n2 in visitados:
                continue
            if glifos1 == glifos2:
                nodo1, nodo2 = G.nodes[n1], G.nodes[n2]
                if (
                    abs(_as_float(nodo1.get("θ", 0.0)) - _as_float(nodo2.get("θ", 0.0))) < PHASE_GROUP_TOL
                    and abs(_as_float(nodo1.get("EPI", 0.0)) - _as_float(nodo2.get("EPI", 0.0))) < umbral_coherencia
                ):
                    grupo.append(n2)
        if len(grupo) >= 4:
            grupos.append(grupo)
            visitados.update(grupo)

    log_macros: List[Dict[str, Any]] = []
    nuevos_nodos: List[Tuple[str, Dict[str, Any]]] = []
    conexiones: List[Tuple[str, Any]] = []

    for idx, grupo in enumerate(grupos):
        # Glifo predominante del grupo
        glifos_grupo: List[str] = []
        for nodo in grupo:
            glifos_grupo += [str(g) for _, g in historial_glifos_por_nodo.get(nodo, [])]
        glifo_predominante = (
            max(set(glifos_grupo), key=glifos_grupo.count) if glifos_grupo else "X"
        )

        # EPI media categorizada
        macro_epi = float(np.mean([_as_float(G.nodes[n].get("EPI", 1.0), 1.0) for n in grupo]))
        if macro_epi > 2.0:
            epi_cat = "H"
        elif macro_epi > 1.2:
            epi_cat = "M"
        else:
            epi_cat = "L"

        nombre_macro = f"M_{glifo_predominante}_{epi_cat}_{idx:02d}"

        macro_vf = float(np.mean([_as_float(G.nodes[n].get("νf", 1.0), 1.0) for n in grupo]))
        macro_Si = float(np.mean([_as_float(G.nodes[n].get("Si", 0.5), 0.5) for n in grupo]))
        macro_theta = float(np.mean([_as_float(G.nodes[n].get("θ", 0.0), 0.0) for n in grupo]))

        nuevo_id = f"{nombre_macro}_N"
        nuevos_nodos.append(
            (
                nuevo_id,
                {
                    "EPI": macro_epi,
                    "νf": macro_vf,
                    "Si": macro_Si,
                    "θ": macro_theta,
                    "ΔNFR": 0.01,
                    "glifo": "NAV",
                    "estado": "activo",
                    "macro": nombre_macro,
                    "categoria": glifo_categoria.get("NAV", "ninguna"),
                },
            )
        )

        # Marcar subnodos y registrar REMESH sin duplicar luego
        for nodo_id in grupo:
            historial_glifos_por_nodo.setdefault(nodo_id, []).append((paso, "REMESH"))
            G.nodes[nodo_id]["_marcar_para_remover"] = True

        # Historial inicial del macronodo
        historial_glifos_por_nodo[nuevo_id] = [
            (paso, "REMESH"),
            (paso, "UM"),
            (paso, "THOL"),
        ]

        # Conexiones: acople por fase/EPI/νf/Si (al menos 3 condiciones)
        for otro in list(G.nodes):
            if otro == nuevo_id:
                continue
            if G.nodes[otro].get("_marcar_para_remover"):
                continue
            nodo_o = G.nodes[otro]
            condiciones = [
                abs(_as_float(nodo_o.get("θ", 0.0)) - macro_theta) < PHASE_GROUP_TOL,
                abs(_as_float(nodo_o.get("EPI", 0.0)) - macro_epi) < EPI_GROUP_TOL_HARD,
                abs(_as_float(nodo_o.get("νf", 0.0)) - macro_vf) < 0.15,
                abs(_as_float(nodo_o.get("Si", 0.0)) - macro_Si) < 0.2,
            ]
            if sum(bool(c) for c in condiciones) >= 3:
                conexiones.append((nuevo_id, otro))

        log_macros.append(
            {
                "entidad": nombre_macro,
                "paso": G.graph.get("paso_actual", paso),
                "nodo": nuevo_id,
                "EPI": round(macro_epi, 3),
                "νf": round(macro_vf, 3),
                "Si": round(macro_Si, 3),
                "θ": round(macro_theta, 3),
                "subnodos": grupo,
            }
        )

    # Integrar EPI compuestas (una sola entrada por registro)
    for entrada in epi_compuestas:
        historial_macronodos.append(
            {
                "paso": entrada.get("paso", paso),
                "glifo": entrada.get("glifo", "REMESH"),
                "miembros": entrada.get("nodos", []),
            }
        )

    # Remover subnodos fusionados
    for n_id in list(G.nodes):
        if G.nodes[n_id].get("_marcar_para_remover"):
            G.remove_node(n_id)

    # Agregar macronodos
    for nuevo_id, attr in nuevos_nodos:
        G.add_node(nuevo_id, **attr)

    # Conectar macronodos
    for a, b in conexiones:
        if a in G.nodes and b in G.nodes:
            G.add_edge(a, b)

    # Asegurar atributos mínimos
    atributos_defecto = {
        "estado": "latente",
        "EPI": 1.0,
        "νf": 1.0,
        "Si": 0.5,
        "θ": 0.0,
        "ΔNFR": 0.0,
        "glifo": "NAV",
        "categoria": "ninguna",
    }

    for n in G.nodes:
        for k, v in atributos_defecto.items():
            if k not in G.nodes[n]:
                G.nodes[n][k] = v

    macronodes_info = {"nodos": [nid for nid, _ in nuevos_nodos], "conexiones": conexiones}

    return historial_macronodos, macronodes_info


def algo_se_mueve(G: Any, historial_glifos_por_nodo: Dict[Any, List[Tuple[int, str]]], paso: int, umbral: float = 0.01) -> bool:
    """Chequeo rápido de actividad: estado activo, cambio de EPI o variación glífica reciente."""
    for nodo in G.nodes:
        datos = G.nodes[nodo]

        if datos.get("estado") == "activo":
            return True  # hay actividad

        # Cambio reciente de EPI
        epi_actual = _as_float(datos.get("EPI", 0.0), 0.0)
        epi_anterior = _as_float(datos.get("EPI_prev", epi_actual), epi_actual)
        if abs(epi_actual - epi_anterior) > umbral:
            return True

        # Variación glífica en la ventana reciente
        historial = historial_glifos_por_nodo.get(nodo, [])
        if len(historial) >= 5:
            glifos_ultimos = [str(g) for _, g in historial[-5:]]
            if len(set(glifos_ultimos)) > 1:
                return True

    return False


def extraer_dinamica_si(G_historia: Iterable[Any]) -> List[List[Dict[str, Any]]]:
    """Devuelve la historia temporal de Si para cada nodo en cada paso."""
    historia_si: List[List[Dict[str, Any]]] = []
    for paso, G in enumerate(G_historia):
        paso_data: List[Dict[str, Any]] = []
        for n in G.nodes:
            paso_data.append({"nodo": n, "paso": paso, "Si": round(_as_float(G.nodes[n].get("Si", 0.0), 0.0), 3)})
        historia_si.append(paso_data)
    return historia_si


def evaluar_si_nodal(nodo: Dict[str, Any], paso: int | None = None) -> float:
    """Actualiza y devuelve Si (índice de sentido) del nodo según TNFR.

    Modelo operativo: Si ≈ (νf / (1 + |ΔNFR|)) * k(glifo) * θ
    - θ se interpreta como fase [0,1] (se clampéa a ese rango por seguridad).
    - k(glifo) pondera la función estructural (IL>UM>VAL>...>SHA),
      conforme a la tabla operativa TNFR.
    """
    # Factores base
    vf = _as_float(nodo.get("νf", 1.0), 1.0)
    dNFR = _as_float(nodo.get("ΔNFR", 0.0), 0.0)
    theta = _as_float(nodo.get("θ", 0.5), 0.5)
    # clamp de fase
    theta = max(0.0, min(1.0, theta))

    glifo = str(nodo.get("glifo", "ninguno")).upper()

    # Pesos glífico-estructurales (coherentes con la tabla operativa)
    pesos_glifo = {
        "AL": 1.0,
        "EN": 1.1,
        "IL": 1.3,
        "OZ": 0.6,
        "UM": 1.2,
        "RA": 1.5,
        "SHA": 0.4,
        "VAL": 1.4,
        "NUL": 0.8,
        "THOL": 1.6,
        "ZHIR": 1.7,
        "NAV": 1.0,
        "REMESH": 1.3,
        "NINGUNO": 1.0,
    }
    k_glifo = float(pesos_glifo.get(glifo, 1.0))

    # Cálculo resonante de Si
    Si_nuevo = round((vf / (1 + abs(dNFR))) * k_glifo * theta, 3)

    # Persistencia
    nodo["Si"] = Si_nuevo
    if paso is not None:
        nodo.setdefault("historial_Si", []).append((paso, Si_nuevo))

    return Si_nuevo


def reciente_glifo(
    nodo_id: Any,
    glifo_objetivo: str,
    historial: Dict[Any, List[Tuple[int, str]]],
    pasos: int = 5,
) -> bool:
    """Comprueba si un glifo objetivo ocurrió en los últimos `pasos` eventos para el nodo.
    Robusto a historiales con códigos numéricos (se normaliza a string).
    """
    eventos = historial.get(nodo_id, [])
    if not eventos:
        return False
    try:
        ultimo_paso = int(eventos[-1][0])
    except (ValueError, TypeError):
        return False

    glifo_str = str(glifo_objetivo).upper()
    ventana = eventos[-(pasos + 1) :]
    for p, g in ventana:
        try:
            p_int = int(p)
        except (ValueError, TypeError):
            continue
        if p_int >= ultimo_paso - pasos and str(g).upper() == glifo_str:
            return True
    return False


def obtener_nodos_emitidos(G: Any):
    """Devuelve (lista_nodos, detalle_por_nodo) para nodos con glifo/categoría armónica.
    Considera emitidos los nodos con glifo distinto de "ninguno" y categoría válida.
    """
    if len(G.nodes) == 0:
        return [], []

    emitidos_final = [
        n
        for n in G.nodes
        if str(G.nodes[n].get("glifo", "ninguno")).lower() != "ninguno"
        and G.nodes[n].get("categoria", "ninguna") not in ["sin categoría", "ninguna"]
    ]

    resultado_detallado = []
    for n in emitidos_final:
        nodo = G.nodes[n]
        entrada = {
            "nodo": n,
            "glifo": nodo.get("glifo"),
            "EPI": round(_as_float(nodo.get("EPI", 0.0), 0.0), 4),
            "Si": round(_as_float(nodo.get("Si", 0.0), 0.0), 4),
            "ΔNFR": round(_as_float(nodo.get("ΔNFR", 0.0), 0.0), 4),
            "θ": round(_as_float(nodo.get("θ", 0.0), 0.0), 4),
            "νf": round(_as_float(nodo.get("νf", 1.0), 1.0), 4),
            "categoria": nodo.get("categoria", "ninguna"),
        }
        resultado_detallado.append(entrada)

    return emitidos_final, resultado_detallado


def exportar_nodos_emitidos(G: Any, emitidos_final: List[Any] | None = None, archivo: str = "nodos_emitidos.json") -> Dict[str, Any]:
    """Prepara metadatos para exportación de nodos emitidos. (No escribe archivo aquí)."""
    try:
        if emitidos_final is None:
            emitidos_final, _ = obtener_nodos_emitidos(G)

        if not emitidos_final:
            return {"exitosa": False, "razon": "No hay nodos emitidos para exportar", "nodos_exportados": 0}

        return {"exitosa": True, "archivo": archivo, "nodos_exportados": len(emitidos_final)}

    except Exception as e:
        return {"exitosa": False, "razon": f"Error durante exportación: {str(e)}", "nodos_exportados": 0}


def crear_diccionario_nodos_emitidos(emitidos_final: Iterable[Any]) -> Dict[Any, bool]:
    """Crea un diccionario {nodo: True} para acceso O(1) a pertenencia."""
    return {n: True for n in emitidos_final}


__all__ = [
    "emergencia_nodal",
    "promover_emergente",
    "detectar_nodos_pulsantes",
    "detectar_macronodos",
    "obtener_nodos_emitidos",
    "evaluar_si_nodal",
    "reciente_glifo",
    "algo_se_mueve",
    "extraer_dinamica_si",
    "exportar_nodos_emitidos",
    "crear_diccionario_nodos_emitidos",
]
