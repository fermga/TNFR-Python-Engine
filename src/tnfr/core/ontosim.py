from typing import List
import networkx as nx
from dataclasses import dataclass
from collections import Counter, deque
from tqdm import tqdm
import math
from math import isnan
import numpy as np
import random

from constants import glifo_categoria
from dynamics import (
    gestionar_conexiones_canonico,
    inicializar_coordinador_temporal_canonico,
    BifurcationManagerTNFR,
    integrar_bifurcaciones_canonicas_en_simulacion,
    integrar_tiempo_topologico_en_simulacion,
    evaluar_activacion_glifica_dinamica,
    limpiar_bifurcaciones_obsoletas,
)
from operators import (
    acoplar_nodos,
    aplicar_remesh_si_estabilizacion_global,
    detectar_EPIs_compuestas,
    glifo_por_estructura,
    aplicar_glifo,
    transicion_glifica_canonica,
    interpretar_sintaxis_glifica,
)
from helpers import (
    evaluar_si_nodal,
    emergencia_nodal,
    detectar_macronodos,
    algo_se_mueve,
    reciente_glifo,
    detectar_nodos_pulsantes,
    promover_emergente,
)


# =============================================================
# Inicialización canónica de NFR (θ = fase estructural)
# =============================================================

def inicializar_nfr_emergente(forma_base, campo_coherencia=None):
    """
    Inicializa un nodo NFR siguiendo condiciones de emergencia TNFR.
    - θ es FASE estructural (sincronía relativa con el campo): θ ∈ [0,1], 1 = máxima sincronía.
    - Si se estima con (EPI, νf, θ) sin usar 'fase' separado.
    """
    if not cumple_condiciones_emergencia(forma_base, campo_coherencia):
        return None

    # Parámetros base
    EPI = evaluar_coherencia_estructural(forma_base)
    νf = calcular_frecuencia_resonante(forma_base)
    Wi_t = generar_matriz_coherencia(forma_base)

    # Estabilidad interna a partir de Wi_t
    estabilidad_interna = float(np.trace(Wi_t) / max(len(Wi_t), 1))

    # θ desde el campo (coherencia de frecuencia con el entorno)
    θ = sincronizar_con_campo(campo_coherencia, νf)  # devuelve [0,1], 1 = sincronía

    # Gradiente nodal aproximado (menor si hay alta estabilidad)
    ΔNFR = round((1.0 - estabilidad_interna) * 0.5 - 0.1, 3)  # rango típico [-0.1, 0.4]

    # Índice de sentido (Si) canónico: aumenta con νf, baja |ΔNFR| y mayor θ
    Si = round((EPI / 2.5) * (νf / 3.0) * θ, 3)

    nfr = {
        "estado": "activo",
        "glifo": "ninguno",
        "categoria": "ninguna",
        "EPI": EPI,
        "EPI_prev": EPI,
        "EPI_prev2": EPI,
        "EPI_prev3": EPI,
        "νf": νf,
        "ΔNFR": ΔNFR,
        "Si": Si,
        "θ": θ,              # θ = fase estructural (coherencia)
        "Wi_t": Wi_t,
        "simetria_interna": round(estabilidad_interna, 3),
    }
    return nfr


# =============================================================
# Construcción de red
# =============================================================

def crear_red_desde_datos(datos: List[dict]) -> nx.Graph:
    G = nx.Graph()
    campo_coherencia = {}

    for nodo_data in datos:
        nodo_id = nodo_data.get('id', f"nodo_{len(G)}")

        if 'forma_base' in nodo_data:
            nfr = inicializar_nfr_emergente(nodo_data['forma_base'], campo_coherencia)
            if nfr:
                G.add_node(nodo_id, **nfr)
                campo_coherencia[nodo_id] = nfr
        else:
            G.add_node(nodo_id, **nodo_data)

    gestionar_conexiones_canonico(G, 0, [])  # inicializa conectividad
    return G


def crear_red_desde_datos_con_barra(datos: list) -> nx.Graph:
    """Igual que crear_red_desde_datos, con barra de progreso."""
    G = nx.Graph()
    campo_coherencia = {}

    print("Inicializando nodos TNFR...")
    for nodo_data in tqdm(datos, desc="Nodos procesados"):
        nodo_id = nodo_data.get('id', f"nodo_{len(G)}")
        if 'forma_base' in nodo_data:
            nfr = inicializar_nfr_emergente(nodo_data['forma_base'], campo_coherencia)
            if nfr:
                G.add_node(nodo_id, **nfr)
                campo_coherencia[nodo_id] = nfr
        else:
            G.add_node(nodo_id, **nodo_data)

    gestionar_conexiones_canonico(G, 0, [])
    return G


# =============================================================
# Heurísticas TNFR locales
# =============================================================

def calcular_frecuencia_resonante(forma_base):
    """
    Determina νf por patrones vibratorios estructurales:
    - Alternancia (vocal/consonante)
    - Densidad energética (oclusivas/continuas/fluidas)
    - Fluidez topológica (transiciones suaves)
    """
    if not forma_base:
        return 1.0

    forma_norm = forma_base.lower()
    longitud = len(forma_norm)

    vocales = "aeiouáéíóúü"
    oclusivas = "pbtdkgqc"
    continuas = "fvszjlmnr"
    fluidas = "wyh"

    alternancias = sum(
        1 for i in range(longitud - 1)
        if (forma_norm[i] in vocales) != (forma_norm[i + 1] in vocales)
    )
    factor_alternancia = alternancias / max(longitud - 1, 1)

    densidad_oclusiva = sum(1 for c in forma_norm if c in oclusivas) / longitud
    densidad_continua = sum(1 for c in forma_norm if c in continuas) / longitud
    densidad_fluida = sum(1 for c in forma_norm if c in fluidas) / longitud

    factor_energia = -0.5 * densidad_oclusiva + 0.3 * densidad_continua + 0.7 * densidad_fluida

    def categoria_fonetica(c):
        if c in vocales: return 'V'
        if c in oclusivas: return 'O'
        if c in continuas: return 'C'
        if c in fluidas: return 'F'
        return 'X'

    transiciones_suaves = sum(
        1 for i in range(longitud - 1)
        if (categoria_fonetica(forma_norm[i]), categoria_fonetica(forma_norm[i + 1]))
           in [('V','C'),('C','V'),('C','F'),('F','V'),('V','F'),('F','C')]
    )
    factor_fluidez = transiciones_suaves / max(longitud - 1, 1)

    freq_base = 1.2 - min(0.4, longitud / 20)
    νf = freq_base * (1.0 + 0.4 * factor_alternancia + 0.3 * factor_energia + 0.3 * factor_fluidez)
    return round(max(0.1, min(3.0, νf)), 3)


def _deben_conectarse_canonico(n1: dict, n2: dict) -> bool:
    """Criterio simple: |Δνf| pequeño y |Δθ| pequeño (θ es fase ∈ [0,1])."""
    phi = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618
    diferencia_vf = abs(n1.get('νf', 1) - n2.get('νf', 1))
    diferencia_theta = abs(n1.get('θ', 0) - n2.get('θ', 0))
    return (diferencia_vf < 0.01 * phi) and (diferencia_theta < 0.25)


# =============================================================
# Simulación canónica
# =============================================================

def simular_emergencia(
    G,
    pasos=100,
    *,
    dt: float = 1.0,
    ruido_sigma: float = 0.0,
    ruido_modo: str = "mult",   # "mult" | "add"
    ruido_dependiente_oz: bool = True,
    k_oz: float = 0.1,
    max_glifos_por_paso: int = 2,
):
    """
    Simulación TNFR:
    - Integra dEPI/dt = νf · ΔNFR
    - Opción de micro-disonancia O'Z ~ sqrt(dt)
    - Límite de glifos por nodo/paso para evitar cascadas
    """

    umbrales = {
        'θ_min': 0.18,
        'EPI_max_dinamico': 3.0,
        'θ_mutacion': 0.25,
        'θ_colapso': 0.45,  # reservado (no se usa como gradiente)
        'bifurcacion_aceleracion': 0.15,
        'EPI_min_coherencia': 0.4,
        'θ_conexion': 0.12,
        'EPI_conexion': 1.8,
        'νf_conexion': 0.2,
        'Si_conexion': 0.25,
        'θ_autoorganizacion': 0.35,
        'bifurcacion_gradiente': 0.8,
        'sensibilidad_calculada': 1.0,
        'factor_densidad': 1.0,
        'fase': 'emergencia',
    }

    global historia_Ct
    if 'historia_Ct' not in globals():
        historia_Ct = []

    historia_epi = []
    historia_glifos = ["paso,nodo,glifo"]
    historial_glifos_por_nodo = {}
    G_historia = []
    registro_conexiones = []

    coordinador_temporal = inicializar_coordinador_temporal_canonico()
    bifurcation_manager = BifurcationManagerTNFR()
    historial_temporal = []

    # Activación mínima inicial
    if all(G.nodes[n]["estado"] in ["latente", "silencio"] for n in G.nodes):
        for n in G.nodes:
            if G.nodes[n]["EPI"] > 0.8 and G.nodes[n]["νf"] > 0.5:
                G.nodes[n]["estado"] = "activo"
                G.nodes[n]["glifo"] = "AL"
                break

    for paso in range(pasos):
        nodos_activos = [n for n in G.nodes if G.nodes[n]["estado"] == "activo"]
        paso_data = []

        acoplar_nodos(G)

        # Umbrales adaptativos
        vf_values = [G.nodes[n]["νf"] for n in G.nodes if G.nodes[n]["estado"] == "activo"]
        dNFR_values = [G.nodes[n]["ΔNFR"] for n in G.nodes if G.nodes[n]["estado"] == "activo"]
        media_vf = np.mean(vf_values) if vf_values else 0
        std_dNFR = np.std(dNFR_values) if dNFR_values else 0

        for n in list(G.nodes):
            nodo = G.nodes[n]

            # Historiales previos
            def valor_valido(x):
                return x is not None and not isinstance(x, str) and not isnan(x)
            for clave in ["EPI_prev", "EPI_prev2", "EPI_prev3"]:
                if not valor_valido(nodo.get(clave)):
                    nodo[clave] = nodo.get("EPI", 1.0)

            glifos_aplicados = 0

            # Dinámica basal
            if nodo["estado"] == "activo":
                nodo["ΔNFR"] = 0.98 * nodo["ΔNFR"] + random.uniform(-0.02, 0.02) * (1.0 - nodo.get("Si", 0))
                nodo["ΔNFR"] = max(min(nodo["ΔNFR"], 1.5), -1.5)
                if nodo["EPI"] < 0.85 and abs(nodo["ΔNFR"]) > 0.4 and nodo["Si"] < 0.3:
                    nodo["estado"] = "inactivo"

            evaluar_si_nodal(nodo, paso)

            # NAV: del silencio a la activación (ΔNFR≈νf)
            if (
                nodo["estado"] == "silencio"
                and abs(nodo["ΔNFR"] - nodo["νf"]) < 0.05
                and nodo.get("Si", 0) > 0.25
                and nodo.get("d2EPI_dt2", 0) > 0.03
                and not reciente_glifo(n, "NAV", historial_glifos_por_nodo, pasos=6)
                and glifos_aplicados < max_glifos_por_paso
            ):
                aplicar_glifo(G, nodo, n, "NAV", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},NAV")
                nodo["estado"] = "activo"
                glifos_aplicados += 1

            # SHA: silencio protector
            if (
                nodo["EPI"] < 0.6 and abs(nodo["ΔNFR"]) > 0.75 and nodo["Si"] < 0.25
                and not reciente_glifo(n, "SHA", historial_glifos_por_nodo, pasos=6)
                and glifos_aplicados < max_glifos_por_paso
            ):
                aplicar_glifo(G, nodo, n, "SHA", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},SHA")
                glifos_aplicados += 1

            # EN: apertura
            if (
                nodo["estado"] == "latente" and abs(nodo["ΔNFR"]) < 0.05 and nodo["Si"] > 0.3
                and not reciente_glifo(n, "EN", historial_glifos_por_nodo, pasos=10)
                and glifos_aplicados < max_glifos_por_paso
            ):
                aplicar_glifo(G, nodo, n, "EN", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},EN")
                glifos_aplicados += 1

            # RA: propagación desde IL
            if (
                nodo.get("glifo") == "IL" and nodo["Si"] > 0.55 and nodo["νf"] > 1.25 and abs(nodo["ΔNFR"]) < 0.15
                and not reciente_glifo(n, "RA", historial_glifos_por_nodo, pasos=8)
                and glifos_aplicados < max_glifos_por_paso
            ):
                aplicar_glifo(G, nodo, n, "RA", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},RA")
                glifos_aplicados += 1

            # UM: acoplamiento por fase
            vecinos = list(G.neighbors(n))
            if (
                nodo["estado"] == "activo" and vecinos
                and sum(1 for v in vecinos if abs(G.nodes[v]["θ"] - nodo["θ"]) < 0.08) >= 2
                and not reciente_glifo(n, "UM", historial_glifos_por_nodo, pasos=8)
                and glifos_aplicados < max_glifos_por_paso
            ):
                aplicar_glifo(G, nodo, n, "UM", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},UM")
                glifos_aplicados += 1

            # ZHIR: mutación por aceleración
            if (
                abs(nodo.get("d2EPI_dt2", 0)) > 0.25 and nodo["Si"] > 0.6
                and not reciente_glifo(n, "ZHIR", historial_glifos_por_nodo, pasos=10)
                and glifos_aplicados < max_glifos_por_paso
            ):
                aplicar_glifo(G, nodo, n, "ZHIR", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},ZHIR")
                glifos_aplicados += 1

            # Evaluación glífica basada en estructura + dinámica
            if emergencia_nodal(nodo, media_vf, std_dNFR):
                glifo = glifo_por_estructura(nodo, G)
                if glifo and glifos_aplicados < max_glifos_por_paso:
                    aplicar_glifo(G, nodo, n, glifo, historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{n},{glifo}")
                    nodo["categoria"] = glifo_categoria.get(glifo, "ninguna")
                    glifos_aplicados += 1

                    vecinos_data = [G.nodes[v] for v in G.neighbors(n)]
                    glifo_dinamico = evaluar_activacion_glifica_dinamica(nodo, umbrales, vecinos_data)
                    if (
                        glifo_dinamico
                        and not reciente_glifo(n, glifo_dinamico, historial_glifos_por_nodo, pasos=8)
                        and glifos_aplicados < max_glifos_por_paso
                    ):
                        aplicar_glifo(G, nodo, n, glifo_dinamico, historial_glifos_por_nodo, paso)
                        historia_glifos.append(f"{paso},{n},{glifo_dinamico}")
                        glifos_aplicados += 1

                    glifo_siguiente = transicion_glifica_canonica(nodo)
                    if glifo_siguiente and glifos_aplicados < max_glifos_por_paso:
                        aplicar_glifo(G, nodo, n, glifo_siguiente, historial_glifos_por_nodo, paso)
                        historia_glifos.append(f"{paso},{n},{glifo_siguiente}")
                        nodo["glifo"] = glifo_siguiente
                        nodo["categoria"] = glifo_categoria.get(glifo_siguiente, "ninguna")
                        glifos_aplicados += 1

            # Activación de VAL controlada
            if (
                nodo["Si"] > 0.8 and nodo["EPI"] > 1.2 and abs(nodo["ΔNFR"]) < 0.2 and nodo.get("dEPI_dt", 0) > 0.15
                and not reciente_glifo(n, "VAL", historial_glifos_por_nodo, pasos=10)
                and glifos_aplicados < max_glifos_por_paso
            ):
                if "expansiones_val" not in nodo:
                    nodo["expansiones_val"] = 0
                if nodo["expansiones_val"] < 3:
                    activar_val_si_estabilidad(n, G, paso, historial_glifos_por_nodo)
                    nodo["expansiones_val"] += 1
                else:
                    aplicar_glifo(G, nodo, n, "THOL", historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{n},THOL")
                    nodo["expansiones_val"] = 0
                glifos_aplicados += 1

            # Contracción desde VAL → NUL si se estabiliza demasiado
            if nodo.get("glifo") == "VAL":
                condiciones_contraccion = (
                    abs(nodo.get("d2EPI_dt2", 0)) < 0.05
                    and abs(nodo.get("ΔNFR", 0)) < 0.1
                    and nodo.get("νf", 1.0) < 1.0
                    and abs(nodo.get("EPI", 0) - nodo.get("EPI_prev", 0)) < 0.01
                )
                if condiciones_contraccion and glifos_aplicados < max_glifos_por_paso:
                    aplicar_glifo(G, nodo, n, "NUL", historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{n},NUL")
                    nodo["glifo"] = "NUL"
                    nodo["categoria"] = glifo_categoria.get("NUL", "ninguna")
                    nodo["expansiones_val"] = 0
                    glifos_aplicados += 1

            # Registro EPI para el paso
            paso_data.append({"nodo": n, "paso": paso, "EPI": round(nodo["EPI"], 2)})

            # Avance de historial EPI
            nodo["EPI_prev3"] = nodo.get("EPI_prev2", nodo["EPI_prev"])
            nodo["EPI_prev2"] = nodo.get("EPI_prev", nodo["EPI"]) 
            nodo["EPI_prev"] = nodo["EPI"] if np.isfinite(nodo["EPI"]) else 1.0

            # Ecuación nodal
            dEPI_dt = nodo["νf"] * nodo["ΔNFR"]
            nodo["EPI"] += dt * dEPI_dt

            # Micro-disonancia opcional
            if ruido_sigma and ruido_sigma > 0:
                sigma = ruido_sigma * (dt ** 0.5)
                if ruido_dependiente_oz:
                    exceso_oz = max(0.0, nodo["ΔNFR"] - nodo["νf"])  # O'Z si ΔNFR > νf
                    sigma *= (1.0 + k_oz * exceso_oz / (nodo["νf"] + 1e-9))
                eta = np.random.normal(0.0, sigma)
                if ruido_modo == "mult":
                    nodo["EPI"] *= (1.0 + eta)
                else:
                    nodo["EPI"] += eta * umbrales['EPI_max_dinamico'] * 0.01

            # Saturación
            EPI_max = umbrales['EPI_max_dinamico']
            nodo["EPI"] = 0.0 if nodo["EPI"] < 0.0 else min(nodo["EPI"], EPI_max)

            # Historiales derivados
            nodo.setdefault("historial_vf", deque(maxlen=512)).append((paso, nodo["νf"]))
            nodo.setdefault("historial_dNFR", deque(maxlen=512)).append((paso, nodo["ΔNFR"]))
            nodo["dEPI_dt"] = dEPI_dt
            nodo.setdefault("historial_dEPI_dt", deque(maxlen=512)).append((paso, dEPI_dt))

            if all(np.isfinite([nodo.get("EPI", 0), nodo.get("EPI_prev", 0), nodo.get("EPI_prev2", 0)])):
                aceleracion = (nodo["EPI"] - 2 * nodo["EPI_prev"] + nodo["EPI_prev2"]) / (dt ** 2)
            else:
                aceleracion = 0.0
            nodo["d2EPI_dt2"] = aceleracion
            nodo.setdefault("historial_aceleracion", deque(maxlen=512)).append((paso, aceleracion))

            # Bifurcaciones canónicas
            integrar_bifurcaciones_canonicas_en_simulacion(G, paso, coordinador_temporal, bifurcation_manager)

            # Contracción resonante
            if nodo.get("estado") == "activo":
                aplicar_contraccion_nul(n, G, paso, historial_glifos_por_nodo)

        # Tiempo topológico TNFR
        resultado_temporal = integrar_tiempo_topologico_en_simulacion(G, paso, coordinador_temporal)
        historial_temporal.append(resultado_temporal['estadisticas'])

        # Conectividad canónica
        umbrales, estadisticas_conexiones = gestionar_conexiones_canonico(G, paso, historia_Ct)

        # Coherencia total C(t)
        C_t = sum(G.nodes[n]["EPI"] for n in G.nodes) / len(G)
        historia_Ct.append((paso, C_t))

        historia_epi.append(paso_data)

        G_snapshot = nx.Graph()
        G_snapshot.add_nodes_from([(n, G.nodes[n].copy()) for n in G.nodes])
        G_snapshot.add_edges_from(G.edges)
        G_historia.append(G_snapshot)

        # REMESH local por estabilidad (no duplica, usar operadores para global)
        for nodo_id in list(historial_glifos_por_nodo.keys()):
            glifos = historial_glifos_por_nodo[nodo_id]
            if (
                len(glifos) >= 3
                and glifos[-1][1] == glifos[-2][1] == glifos[-3][1]
                and abs(G.nodes[nodo_id]["EPI"] - G.nodes[nodo_id]["EPI_prev"]) < 0.05
            ):
                aplicar_glifo(G, G.nodes[nodo_id], nodo_id, "REMESH", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{nodo_id},REMESH")

        # REMESH global (sin duplicar historiales)
        aplicar_remesh_si_estabilizacion_global(G, historial_glifos_por_nodo, historia_glifos, paso)

        # EPIs compuestas y macronodos
        epi_compuestas = detectar_EPIs_compuestas(G, umbrales)
        if algo_se_mueve(G, historial_glifos_por_nodo, paso):
            historial_macronodos, macronodes_info = detectar_macronodos(G, historial_glifos_por_nodo, epi_compuestas, paso)
        else:
            macronodes_info = {'nodos': [], 'conexiones': []}

        # Exceso de VAL → reorganización correctiva
        for nodo_id, glifos in historial_glifos_por_nodo.items():
            ultimos = [g for _, g in glifos[-6:]]
            if ultimos.count("VAL") >= 4 and "THOL" not in ultimos and "ZHIR" not in ultimos:
                nodo = G.nodes[nodo_id]
                if nodo["Si"] > 0.5 and abs(nodo["ΔNFR"]) < 0.2:
                    aplicar_glifo(G, nodo, nodo_id, "THOL", historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{nodo_id},THOL")
                    nodo["expansiones_val"] = 0
                else:
                    aplicar_glifo(G, nodo, nodo_id, "ZHIR", historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{nodo_id},ZHIR")
                    nodo["expansiones_val"] = 0

    # Limpieza periódica de bifurcaciones
    if paso % 300 == 0:
        limpiar_bifurcaciones_obsoletas(bifurcation_manager, paso)

    lecturas = interpretar_sintaxis_glifica(historial_glifos_por_nodo)

    # Diagnóstico simbólico final
    diagnostico = []
    for nodo in G.nodes:
        nombre = nodo
        datos = G.nodes[nodo]
        glifos_nodo = [g[1] for g in historial_glifos_por_nodo.get(nombre, [])]
        mutó = "ZHIR" in glifos_nodo
        en_epi = any(nombre in grupo["nodos"] for grupo in epi_compuestas)
        lectura = lecturas.get(nombre, {}).get("trayectoria", [])
        diagnostico.append({
            "palabra": nombre,
            "glifos": glifos_nodo,
            "lectura_sintactica": lectura,
            "mutó": mutó,
            "en_epi_compuesta": en_epi,
            "Si": datos.get("Si", 0),
            "estado": datos.get("estado", "latente"),
            "categoría": datos.get("categoria", "sin categoría"),
        })

    # Promoción emergente en nodos pulsantes (SIN ruta duplicada)
    for nodo_id in detectar_nodos_pulsantes(historial_glifos_por_nodo):
        if nodo_id in G:
            promover_emergente(nodo_id, G, paso, historial_glifos_por_nodo, historia_glifos)

    bifurcation_stats = bifurcation_manager.obtener_estadisticas_bifurcacion()
    return (
        historia_epi,
        G,
        epi_compuestas,
        lecturas,
        G_historia,
        historial_glifos_por_nodo,
        historial_temporal,
        bifurcation_stats,
    )


# =============================================================
# Acciones glíficas auxiliares
# =============================================================

def aplicar_contraccion_nul(nodo_id, G, paso, historial_glifos_por_nodo):
    nodo = G.nodes[nodo_id]
    condiciones = (
        nodo.get("Si", 1.0) < 0.3
        and abs(nodo.get("ΔNFR", 0.0)) > 0.8
        and nodo.get("estado") == "activo"
        and nodo.get("d2EPI_dt2", 0) < -0.05
    )
    if not condiciones:
        return False

    nodo["EPI"] = round(nodo["EPI"] * 0.7, 3)
    nodo["estado"] = "latente"
    nodo["glifo"] = "NUL"
    nodo["categoria"] = "contractivo"
    historial_glifos_por_nodo.setdefault(nodo_id, []).append((paso, "NUL"))
    nodo["expansiones_val"] = 0
    return True


def activar_val_si_estabilidad(nodo_id, G, paso, historial_glifos_por_nodo):
    nodo = G.nodes[nodo_id]
    if nodo.get("expansiones_val", 0) >= 3:
        return None
    condiciones = (
        nodo.get("Si", 0) > 0.85
        and abs(nodo.get("ΔNFR", 0)) < 0.2
        and nodo.get("dEPI_dt", 0) > 0.18
        and nodo.get("d2EPI_dt2", 0) > 0.2
        and nodo.get("estado") == "activo"
    )
    if not condiciones:
        return None

    nuevo_id = f"{nodo_id}_VAL_{random.randint(1000, 9999)}"
    if nuevo_id in G:
        return None

    nuevo_nodo = {
        "EPI": round(nodo["EPI"] * random.uniform(1.0, 1.1), 3),
        "EPI_prev": nodo["EPI"],
        "EPI_prev2": nodo.get("EPI_prev", nodo["EPI"]),
        "EPI_prev3": nodo.get("EPI_prev2", nodo["EPI"]),
        "glifo": "VAL",
        "categoria": "expansivo",
        "estado": "activo",
        "νf": round(nodo["νf"] * random.uniform(1.0, 1.05), 3),
        "ΔNFR": round(nodo["ΔNFR"] * 0.9, 3),
        "θ": round(nodo["θ"] + random.uniform(-0.01, 0.01), 3),
        "Si": nodo["Si"] * 0.98,
        "historial_glifos": [(paso, "VAL")],
        "historial_vf": [(paso, nodo["νf"])],
        "historial_dNFR": [(paso, nodo["ΔNFR"])],
        "historial_dEPI_dt": [(paso, nodo.get("dEPI_dt", 0))],
        "historial_Si": [(paso, nodo["Si"])],
    }

    G.add_node(nuevo_id, **nuevo_nodo)
    G.add_edge(nodo_id, nuevo_id)

    historial_glifos_por_nodo.setdefault(nodo_id, []).append((paso, "VAL"))
    historial_glifos_por_nodo[nuevo_id] = [(paso, "VAL")]

    nodo["expansiones_val"] = nodo.get("expansiones_val", 0) + 1
    return nuevo_id


# =============================================================
# Condiciones y métricas de coherencia
# =============================================================

def cumple_condiciones_emergencia(forma_base, campo_coherencia):
    """
    Condiciones TNFR de emergencia nodal:
    1) νf > 0.3
    2) estructura no degenerada (diversidad)
    3) acoplamiento posible con el campo
    """
    if not forma_base or len(forma_base) < 2:
        return False

    diversidad = len(set(forma_base)) / len(forma_base)
    if diversidad < 0.3:
        return False

    freq_potencial = calcular_frecuencia_resonante(forma_base)
    if freq_potencial < 0.3:
        return False

    if campo_coherencia and len(campo_coherencia) > 0:
        coherencia_promedio = np.mean([nodo.get("EPI", 1.0) for nodo in campo_coherencia.values()])
        if coherencia_promedio > 0 and freq_potencial > coherencia_promedio * 2.5:
            return False

    return True


def evaluar_coherencia_estructural(forma_base):
    """Calcula EPI basado en simetría, diversidad, estabilidad y coherencia fónica."""
    if not forma_base:
        return 1.0

    forma_norm = forma_base.lower()
    longitud = len(forma_norm)

    def calcular_simetria(s):
        centro = len(s) // 2
        if len(s) % 2 == 0:
            izq, der = s[:centro], s[centro:][::-1]
        else:
            izq, der = s[:centro], s[centro+1:][::-1]
        coincidencias = sum(1 for a, b in zip(izq, der) if a == b)
        return coincidencias / max(len(izq), 1)

    simetria = calcular_simetria(forma_norm)
    diversidad = len(set(forma_norm)) / longitud

    contador = Counter(forma_norm)
    entropia = -sum((freq/longitud) * np.log2(freq/longitud) for freq in contador.values())
    estabilidad = min(1.0, entropia / 3.0)

    vocales = "aeiouáéíóúü"
    patron_vocal = sum(1 for c in forma_norm if c in vocales) / longitud
    coherencia_fonetica = min(1.0, abs(0.4 - patron_vocal) * 2.5)

    EPI = 0.3 * simetria + 0.25 * diversidad + 0.25 * estabilidad + 0.2 * coherencia_fonetica
    return round(0.5 + EPI * 2.0, 3)


def generar_matriz_coherencia(forma_base):
    """Crea matriz Wi(t) modelando subnodos internos y acoplamientos."""
    if not forma_base or len(forma_base) < 2:
        return np.array([[1.0]])

    longitud = len(forma_base)
    Wi = np.zeros((longitud, longitud))

    for i in range(longitud - 1):
        Wi[i][i+1] = Wi[i+1][i] = 0.8

    for i in range(longitud):
        for j in range(i+2, longitud):
            if forma_base[i].lower() == forma_base[j].lower():
                Wi[i][j] = Wi[j][i] = 0.3

    np.fill_diagonal(Wi, 1.0)

    for i in range(longitud):
        suma_fila = np.sum(Wi[i])
        if suma_fila > 0:
            Wi[i] = Wi[i] / suma_fila

    return Wi


def sincronizar_con_campo(campo_coherencia, νf_nodo):
    """
    Devuelve **θ** (fase estructural) del nodo respecto al campo de coherencia.
    θ ∈ [0,1], donde 1 = sincronía alta con el campo y 0 = disonancia.
    """
    if not campo_coherencia or len(campo_coherencia) == 0:
        return 1.0  # sin campo: asumimos sincronía neutra alta

    frecuencias_campo = [nodo.get("νf", 1.0) for nodo in campo_coherencia.values()]
    freq_promedio_campo = float(np.mean(frecuencias_campo))
    diferencia_freq = abs(νf_nodo - freq_promedio_campo)

    # Mapear diferencia de frecuencia a θ (inverso a la disonancia)
    if diferencia_freq < 0.1:
        theta = 1.0
    elif diferencia_freq < 0.3:
        theta = 0.75
    elif diferencia_freq < 0.6:
        theta = 0.5
    elif diferencia_freq < 1.0:
        theta = 0.25
    else:
        theta = 0.0

    return round(theta, 3)


__all__ = [
    'inicializar_nfr_emergente',
    '_deben_conectarse_canonico',
    'calcular_frecuencia_resonante',
    'crear_red_desde_datos_con_barra',
    'simular_emergencia',
    'aplicar_contraccion_nul',
    'activar_val_si_estabilidad',
    'cumple_condiciones_emergencia',
    'evaluar_coherencia_estructural',
    'generar_matriz_coherencia',
    'sincronizar_con_campo',
    'gestionar_conexiones_canonico',
]
