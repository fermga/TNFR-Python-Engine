from typing import List
import networkx as nx
from dataclasses import dataclass
from collections import Counter
import math
from math import isnan
import numpy as np
import random
from tnfr.resonance.dynamics import inicializar_coordinador_temporal_canonico
from tnfr.resonance.dynamics import BifurcationManagerTNFR
from tnfr.resonance.dynamics import integrar_bifurcaciones_canonicas_en_simulacion
from tnfr.resonance.dynamics import integrar_tiempo_topologico_en_simulacion
from tnfr.resonance.dynamics import evaluar_activacion_glifica_dinamica
from tnfr.matrix.operators import acoplar_nodos
from tnfr.matrix.operators import aplicar_remesh_si_estabilizacion_global
from tnfr.matrix.operators import detectar_EPIs_compuestas
from tnfr.matrix.operators import glifo_por_estructura
from tnfr.matrix.operators import aplicar_glifo
from tnfr.matrix.operators import transicion_glifica_canonica
from tnfr.matrix.operators import interpretar_sintaxis_glífica
from tnfr.utils.helpers import evaluar_si_nodal
from tnfr.utils.helpers import emergencia_nodal
from tnfr.utils.helpers import detectar_macronodos
from tnfr.utils.helpers import algo_se_mueve
from tnfr.utils.helpers import reciente_glifo
from tnfr.utils.helpers import detectar_nodos_pulsantes
from tnfr.utils.helpers import promover_emergente

def inicializar_nfr_emergente(forma_base, campo_coherencia=None):
    """
    Inicializa NFR siguiendo condiciones de emergencia nodal TNFR.
    
    Reemplaza las heurísticas ad-hoc por evaluación estructural canónica.
    """
    # Verificar condiciones de emergencia
    if not cumple_condiciones_emergencia(forma_base, campo_coherencia):
        return None
    
    # Calcular parámetros estructurales
    EPI = evaluar_coherencia_estructural(forma_base)
    νf = calcular_frecuencia_resonante(forma_base)
    Wi_t = generar_matriz_coherencia(forma_base)
    fase = sincronizar_con_campo(campo_coherencia, νf)
    
    # Calcular parámetros derivados
    # ΔNFR: gradiente nodal basado en estabilidad interna de Wi_t
    estabilidad_interna = np.trace(Wi_t) / len(Wi_t)
    ΔNFR = round((1.0 - estabilidad_interna) * 0.5 - 0.1, 3)  # rango típico [-0.1, 0.4]
    
    # Si: índice de sentido basado en coherencia estructural y frecuencia
    Si = round((EPI / 2.5) * (νf / 3.0) * (1.0 - fase), 3)  # decrece con disonancia
    
    # θ: umbral estructural basado en EPI y estabilidad
    θ = round(min(1.0, EPI * estabilidad_interna * 0.4), 3)
    
    # Crear NFR canónico
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
        "θ": θ,
        "Wi_t": Wi_t,
        "fase": fase,
        "simetria_interna": round(estabilidad_interna, 3)
    }
    
    return nfr

def crear_red_desde_datos(datos: List[dict]) -> nx.Graph:
    """Crea red TNFR desde datos estructurados - NUEVA FUNCIÓN"""
    G = nx.Graph()
    campo_coherencia = {}
    
    for nodo_data in datos:
        nodo_id = nodo_data.get('id', f"nodo_{len(G)}")
        
        # Usar inicialización canónica existente
        if 'forma_base' in nodo_data:
            nfr = inicializar_nfr_emergente(nodo_data['forma_base'], campo_coherencia)
            if nfr:
                G.add_node(nodo_id, **nfr)
                campo_coherencia[nodo_id] = nfr
        else:
            # Datos ya procesados
            G.add_node(nodo_id, **nodo_data)
    
    # Usar conectividad canónica existente  
    umbrales, _ = gestionar_conexiones_canonico(G, 0, [])
    return G

def calcular_frecuencia_resonante(forma_base):
    """
    Determina νf por patrones vibratorios estructurales.
    
    La frecuencia resonante depende de:
    - Alternancia estructural (consonante/vocal)
    - Densidad energética (consonantes oclusivas vs continuas)
    - Fluidez topológica (transiciones suaves)
    """
    if not forma_base:
        return 1.0
    
    forma_norm = forma_base.lower()
    longitud = len(forma_norm)
    
    # Clasificación fonética TNFR
    vocales = "aeiouáéíóúü"
    oclusivas = "pbtdkgqc"      # alta energía, baja frecuencia
    continuas = "fvszjlmnr"     # media energía, alta frecuencia  
    fluidas = "wyh"             # baja energía, muy alta frecuencia
    
    # Factor de alternancia (patrones alternos)
    alternancias = 0
    for i in range(longitud - 1):
        actual = forma_norm[i] in vocales
        siguiente = forma_norm[i+1] in vocales
        if actual != siguiente:  # transición vocal-consonante o viceversa
            alternancias += 1
    
    factor_alternancia = alternancias / max(longitud - 1, 1)
    
    # Factor de densidad energética
    densidad_oclusiva = sum(1 for c in forma_norm if c in oclusivas) / longitud
    densidad_continua = sum(1 for c in forma_norm if c in continuas) / longitud
    densidad_fluida = sum(1 for c in forma_norm if c in fluidas) / longitud
    
    # Las continuas y fluidas aumentan frecuencia, las oclusivas la reducen
    factor_energia = (
        -0.5 * densidad_oclusiva +   # reducen frecuencia
        0.3 * densidad_continua +    # aumentan ligeramente
        0.7 * densidad_fluida        # aumentan significativamente
    )
    
    # Factor de fluidez (transiciones suaves entre fonemas similares)
    def categoria_fonetica(c):
        if c in vocales: return 'V'
        elif c in oclusivas: return 'O'
        elif c in continuas: return 'C'
        elif c in fluidas: return 'F'
        else: return 'X'
    
    transiciones_suaves = 0
    for i in range(longitud - 1):
        cat1 = categoria_fonetica(forma_norm[i])
        cat2 = categoria_fonetica(forma_norm[i+1])
        # Transiciones suaves: V-C, C-V, C-F, F-V
        if (cat1, cat2) in [('V','C'), ('C','V'), ('C','F'), ('F','V'), ('V','F'), ('F','C')]:
            transiciones_suaves += 1
    
    factor_fluidez = transiciones_suaves / max(longitud - 1, 1)
    
    # Frecuencia base según longitud (formas más largas tienden a menor frecuencia)
    freq_base = 1.2 - min(0.4, longitud / 20)
    
    # Combinar todos los factores
    νf = freq_base * (
        1.0 + 
        0.4 * factor_alternancia +   # alternancia aumenta frecuencia
        0.3 * factor_energia +       # balance energético
        0.3 * factor_fluidez         # fluidez aumenta frecuencia
    )
    
    # Limitar al rango válido [0.1, 3.0]
    νf = max(0.1, min(3.0, νf))
    
    return round(νf, 3)

def _deben_conectarse_canonico(n1: dict, n2: dict) -> bool:
    """Mejora la lógica existente con umbral áureo"""
    phi = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618
    
    diferencia_vf = abs(n1.get('νf', 1) - n2.get('νf', 1))
    diferencia_fase = abs(n1.get('fase', 0) - n2.get('fase', 0)) % (2 * math.pi)
    
    return (diferencia_vf < 0.01 * phi and 
            diferencia_fase < math.pi / 2)

def simular_emergencia(G, pasos=100):

    umbrales = {
        'θ_min': 0.18,
        'EPI_max_dinamico': 3.0,
        'θ_mutacion': 0.25,
        'θ_colapso': 0.45,
        'bifurcacion_aceleracion': 0.15,
        'EPI_min_coherencia': 0.4,   # ← Añade este valor por defecto
        'θ_conexion': 0.12,
        'EPI_conexion': 1.8,
        'νf_conexion': 0.2,
        'Si_conexion': 0.25,
        'θ_autoorganizacion': 0.35,
        'bifurcacion_gradiente': 0.8,
        'sensibilidad_calculada': 1.0,
        'factor_densidad': 1.0,
        'fase': 'emergencia'
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

    glifo_categoria = {
        "AL": "activador", "EN": "receptor", "IL": "estabilizador",
        "OZ": "disonante", "UM": "acoplador", "RA": "resonador",
        "SHA": "latente", "VAL": "expansivo", "NUL": "contractivo",
        "THOL": "autoorganizador", "ZHIR": "mutante", "NAV": "transicional",
        "REMESH": "recursivo"
    }

    # Activación mínima inicial si todos están inactivos o silenciosos
    if all(G.nodes[n]["estado"] in ["latente", "silencio"] for n in G.nodes):
        for n in G.nodes:
            if G.nodes[n]["EPI"] > 0.8 and G.nodes[n]["νf"] > 0.5:
                G.nodes[n]["estado"] = "activo"
                G.nodes[n]["glifo"] = "AL"
                break  # activa solo uno, para iniciar pulso

    for paso in range(pasos):
        nodos_activos = [n for n in G.nodes if G.nodes[n]["estado"] == "activo"]
        paso_data = [] 

        acoplar_nodos(G)

        # Cálculo de umbrales adaptativos para emergencia nodal
        vf_values = [G.nodes[n]["νf"] for n in G.nodes if G.nodes[n]["estado"] == "activo"]
        dNFR_values = [G.nodes[n]["ΔNFR"] for n in G.nodes if G.nodes[n]["estado"] == "activo"]

        media_vf = np.mean(vf_values) if vf_values else 0
        std_dNFR = np.std(dNFR_values) if dNFR_values else 0

        for n in list(G.nodes):

            nodo = G.nodes[n]
            def valor_valido(x):
                return x is not None and not isinstance(x, str) and not isnan(x)

            for n in list(G.nodes):
                nodo = G.nodes[n]
                
                for clave in ["EPI_prev", "EPI_prev2", "EPI_prev3"]:
                    if not valor_valido(nodo.get(clave)):
                        nodo[clave] = nodo.get("EPI", 1.0)

            if nodo["estado"] == "activo":
                # Dinámica basal influida por νf y sentido
                factor_ruido = random.uniform(0.98, 1.02) + 0.02 * random.uniform(-1, 1) * (1 - nodo["Si"])
                modulador = factor_ruido * (1 + 0.02 * min(nodo.get("νf", 1.0), 5))  # cap νf por seguridad

                nodo["EPI"] *= modulador

                # Evitar NaN o valores extremos
                if not np.isfinite(nodo["EPI"]) or nodo["EPI"] > 10:
                    nodo["EPI"] = 1.0 + random.uniform(-0.05, 0.05)  # reset suave)
                if nodo["EPI"] > 1e4:
                    nodo["EPI"] = 1e4
                nodo["ΔNFR"] += random.uniform(-0.08, 0.08) * (1.1 - nodo["Si"])
                nodo["ΔNFR"] = max(min(nodo["ΔNFR"], 1.5), -1.5) 

                # Condición de apagado nodal si pierde coherencia estructural
                if (
                    nodo["EPI"] < 0.85
                    and abs(nodo["ΔNFR"]) > 0.4
                    and nodo["Si"] < 0.3
                ):
                    nodo["estado"] = "inactivo"

            evaluar_si_nodal(nodo, paso)

            if (
                nodo["estado"] == "silencio"
                and abs(nodo["ΔNFR"] - nodo["νf"]) < 0.05
                and nodo.get("Si", 0) > 0.25
                and nodo.get("d2EPI_dt2", 0) > 0.03
                and not reciente_glifo(n, "NAV", historial_glifos_por_nodo, pasos=6)
            ):
                aplicar_glifo(G, nodo, n, "NAV", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},NAV")
                nodo["estado"] = "activo"

            if (
                nodo["EPI"] < 0.6
                and abs(nodo["ΔNFR"]) > 0.75
                and nodo["Si"] < 0.25
                and not reciente_glifo(n, "SHA", historial_glifos_por_nodo, pasos=6)
            ):
                aplicar_glifo(G, nodo, n, "SHA", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},SHA")
                continue

            if (
                nodo["estado"] == "latente"
                and abs(nodo["ΔNFR"]) < 0.05
                and nodo["Si"] > 0.3
                and not reciente_glifo(n, "EN", historial_glifos_por_nodo, pasos=10)
            ):
                aplicar_glifo(G, nodo, n, "EN", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},EN")

            if (
                nodo["glifo"] == "IL"
                and nodo["Si"] > 0.55
                and nodo["νf"] > 1.25
                and abs(nodo["ΔNFR"]) < 0.15  # Baja necesidad de reorganización
                and not reciente_glifo(n, "RA", historial_glifos_por_nodo, pasos=8)
            ):
                aplicar_glifo(G, nodo, n, "RA", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},RA")

            vecinos = list(G.neighbors(n))
            if (
                nodo["estado"] == "activo"
                and vecinos
                and sum(1 for v in vecinos if abs(G.nodes[v]["θ"] - nodo["θ"]) < 0.08) >= 2
                and not reciente_glifo(n, "UM", historial_glifos_por_nodo, pasos=8)
            ):
                aplicar_glifo(G, nodo, n, "UM", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},UM")

            if (
                abs(nodo.get("d2EPI_dt2", 0)) > 0.25
                and nodo["Si"] > 0.6
                and not reciente_glifo(n, "ZHIR", historial_glifos_por_nodo, pasos=10)
            ):
                aplicar_glifo(G, nodo, n, "ZHIR", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},ZHIR")

            if emergencia_nodal(nodo, media_vf, std_dNFR):
                glifo = glifo_por_estructura(nodo, G)
                if glifo:
                    aplicar_glifo(G, nodo, n, glifo, historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{n},{glifo}")
                    nodo["categoria"] = glifo_categoria.get(glifo, "ninguna")

                    # Evaluación glífica con umbrales dinámicos (mejora canónica)
                    vecinos_data = [G.nodes[v] for v in G.neighbors(n)]
                    glifo_dinamico = evaluar_activacion_glifica_dinamica(nodo, umbrales, vecinos_data)

                    if glifo_dinamico and not reciente_glifo(n, glifo_dinamico, historial_glifos_por_nodo, pasos=8):
                        aplicar_glifo(G, nodo, n, glifo_dinamico, historial_glifos_por_nodo, paso)
                        historia_glifos.append(f"{paso},{n},{glifo_dinamico}")

                    glifo_siguiente = transicion_glifica_canonica(nodo)
                    if glifo_siguiente:
                        aplicar_glifo(G, nodo, n, glifo_siguiente, historial_glifos_por_nodo, paso)
                        historia_glifos.append(f"{paso},{n},{glifo_siguiente}")
                        nodo["glifo"] = glifo_siguiente
                        nodo["categoria"] = glifo_categoria.get(glifo_siguiente, "ninguna")

            # Activación estructural de VAL (expansión controlada)
            if (
                nodo["Si"] > 0.8
                and nodo["EPI"] > 1.2
                and abs(nodo["ΔNFR"]) < 0.2
                and nodo.get("dEPI_dt", 0) > 0.15
                and not reciente_glifo(n, "VAL", historial_glifos_por_nodo, pasos=10)
            ):
                if "expansiones_val" not in nodo:
                    nodo["expansiones_val"] = 0

                if nodo["expansiones_val"] < 3:
                    activar_val_si_estabilidad(n, G, paso, historial_glifos_por_nodo)
                    nodo["expansiones_val"] += 1
                else:
                    aplicar_glifo(G, nodo, n, "THOL", historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{n},THOL")

            if nodo.get("glifo") == "VAL":
                condiciones_contraccion = (
                    abs(nodo.get("d2EPI_dt2", 0)) < 0.05 and
                    abs(nodo.get("ΔNFR", 0)) < 0.1 and
                    nodo.get("νf", 1.0) < 1.0 and
                    abs(nodo.get("EPI", 0) - nodo.get("EPI_prev", 0)) < 0.01
                )

                if condiciones_contraccion:
                    aplicar_glifo(G, nodo, n, "NUL", historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{n},NUL")
                    nodo["glifo"] = "NUL"
                    nodo["categoria"] = glifo_categoria.get("NUL", "ninguna")

            paso_data.append({
                "nodo": n, 
                "paso": paso, 
                "EPI": round(nodo["EPI"], 2)
            })
            nodo["EPI_prev3"] = nodo.get("EPI_prev2", nodo["EPI_prev"])
            nodo["EPI_prev2"] = nodo.get("EPI_prev", nodo["EPI"])
            nodo["EPI_prev"] = nodo["EPI"] if np.isfinite(nodo["EPI"]) else 1.0

            # Cálculo de ∂EPI/∂t = νf · ΔNFR
            dEPI_dt = nodo["νf"] * nodo["ΔNFR"]
            nodo["dEPI_dt"] = dEPI_dt
            if "historial_dEPI_dt" not in nodo:
                nodo["historial_dEPI_dt"] = []
            nodo["historial_dEPI_dt"].append((paso, dEPI_dt))

            # Registrar evolución de νf y ΔNFR
            if "historial_vf" not in nodo:
                nodo["historial_vf"] = []
            if "historial_dNFR" not in nodo:
                nodo["historial_dNFR"] = []

            nodo["historial_vf"].append((paso, nodo["νf"]))
            nodo["historial_dNFR"].append((paso, nodo["ΔNFR"]))

            # Calcular aceleración estructural ∂²EPI/∂t² solo si los valores son válidos
            if all(np.isfinite([nodo.get("EPI", 0), nodo.get("EPI_prev", 0), nodo.get("EPI_prev2", 0)])):
                aceleracion = nodo["EPI"] - 2 * nodo["EPI_prev"] + nodo["EPI_prev2"]
            else:
                aceleracion = 0.0  # O un valor neutro que no active mutaciones erróneas

            nodo["d2EPI_dt2"] = aceleracion

            # Umbral de bifurcación: si se supera, aplicar THOL
            resultado_bifurcaciones = integrar_bifurcaciones_canonicas_en_simulacion(
                G, paso, coordinador_temporal, bifurcation_manager
            )

            # Evaluar contracción si hay disonancia o colapso de sentido (NU´L)
            if nodo.get("estado") == "activo":
                aplicar_contraccion_nul(n, G, paso, historial_glifos_por_nodo)

            # === CONTROL DE EXPANSIÓN INFINITA ===
            if "expansiones_val" not in nodo:
                nodo["expansiones_val"] = 0

            if nodo["expansiones_val"] >= 3:
                continue  # evita expansión si ya lo hizo demasiadas veces

            # Aquí sí puede expandirse:
            activar_val_si_estabilidad(n, G, paso, historial_glifos_por_nodo)
            nodo["expansiones_val"] += 1

            if (
                nodo.get("estado") == "activo"
                and nodo.get("Si", 0) > 0.8
                and nodo.get("EPI", 0) > 1.1
                and abs(nodo.get("ΔNFR", 0)) < 0.25
                and nodo.get("dEPI_dt", 0) > 0.15
                and not reciente_glifo(n, "VAL", historial_glifos_por_nodo, pasos=8)
            ):
                activar_val_si_estabilidad(n, G, paso, historial_glifos_por_nodo)

            # Guardar aceleración para graficar más tarde
            if "historial_aceleracion" not in nodo:
                nodo["historial_aceleracion"] = []
            nodo["historial_aceleracion"].append((paso, aceleracion))

        # Gestión temporal topológica TNFR
        resultado_temporal = integrar_tiempo_topologico_en_simulacion(G, paso, coordinador_temporal)
        historial_temporal.append(resultado_temporal['estadisticas'])

        # Gestión de conexiones con información temporal
        umbrales, estadisticas_conexiones = gestionar_conexiones_canonico(G, paso, historia_Ct)

        # Calcular coherencia total C(t) al final del paso
        C_t = sum(G.nodes[n]["EPI"] for n in G.nodes) / len(G)
        historia_Ct.append((paso, C_t))

        historia_epi.append(paso_data)

        G_snapshot = nx.Graph()
        G_snapshot.add_nodes_from([(n, G.nodes[n].copy()) for n in G.nodes])
        G_snapshot.add_edges_from(G.edges)
        G_historia.append(G_snapshot)

        for nodo_id in list(historial_glifos_por_nodo.keys()):
            glifos = historial_glifos_por_nodo[nodo_id]

            if (
                len(glifos) >= 3 
                and glifos[-1][1] == glifos[-2][1] == glifos[-3][1]
                and abs(G.nodes[nodo_id]["EPI"] - G.nodes[nodo_id]["EPI_prev"]) < 0.05
            ):
                aplicar_glifo(G, G.nodes[nodo_id], nodo_id, "REMESH", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{nodo_id},REMESH")

        aplicar_remesh_si_estabilizacion_global(G, historial_glifos_por_nodo, historia_glifos, paso)
        aplicar_remesh_grupal(G, historial_glifos_por_nodo)
        epi_compuestas = detectar_EPIs_compuestas(G, umbrales)
        if algo_se_mueve(G, historial_glifos_por_nodo, paso):
            historial_macronodos, macronodes_info = detectar_macronodos(G, historial_glifos_por_nodo, epi_compuestas, paso)
          
        else:
            macronodes_info = {'nodos': [], 'conexiones': []}

        # Evaluar exceso de VAL y promover reorganización estructural
        for nodo_id, glifos in historial_glifos_por_nodo.items():
            ultimos = [g for _, g in glifos[-6:]]  # últimos 6 glifos del nodo
            if ultimos.count("VAL") >= 4 and "THOL" not in ultimos and "ZHIR" not in ultimos:
                nodo = G.nodes[nodo_id]
                
                # Se decide el glifo correctivo en función de su Si y ΔNFR
                if nodo["Si"] > 0.5 and abs(nodo["ΔNFR"]) < 0.2:
                    aplicar_glifo(G, nodo, nodo_id, "THOL", historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{nodo_id},THOL")
                else:
                    aplicar_glifo(G, nodo, nodo_id, "ZHIR", historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{nodo_id},ZHIR")

        nodos_activos = [n for n in G.nodes if G.nodes[n]["estado"] == "activo"]

    # Limpiar bifurcaciones obsoletas cada 300 pasos
    if paso % 300 == 0:
        obsoletas = limpiar_bifurcaciones_obsoletas(bifurcation_manager, paso)
    
    lecturas = interpretar_sintaxis_glífica(historial_glifos_por_nodo)

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
            "categoría": datos.get("categoria", "sin categoría")
        })

    nodos_pulsantes = detectar_nodos_pulsantes(historial_glifos_por_nodo)

    for nodo_id in nodos_pulsantes:
        nodo = G.nodes[nodo_id]
        historial = historial_glifos_por_nodo.get(nodo_id, [])
        ultimos = [g for _, g in historial][-6:]

        if nodo["glifo"] in ["THOL", "ZHIR", "REMESH"]:
            continue  # ya está mutado o recursivo

        nodo = G.nodes[nodo_id]

        # Evaluar emergente canónico
        if abs(nodo["EPI"] - nodo["EPI_prev"]) < 0.01 and abs(nodo["ΔNFR"]) < 0.05:
            glifo = "REMESH"
        elif abs(nodo.get("θ", 0) - nodo.get("θ_prev", 0)) > 0.2:
            glifo = "ZHIR"
        elif nodo.get("Si", 0) > 0.8 and nodo.get("glifo") == "UM":
            glifo = "RA"
        else:
            glifo = "THOL"

    if nodo_id in G:
        promover_emergente(nodo_id, G, paso, historial_glifos_por_nodo, historia_glifos)

    bifurcation_stats = bifurcation_manager.obtener_estadisticas_bifurcacion()
    return historia_epi, G, epi_compuestas, lecturas, G_historia, historial_glifos_por_nodo, historial_temporal, bifurcation_stats

def aplicar_contraccion_nul(nodo_id, G, paso, historial_glifos_por_nodo):
    nodo = G.nodes[nodo_id]

    condiciones = (
        nodo.get("Si", 1.0) < 0.3 and
        abs(nodo.get("ΔNFR", 0.0)) > 0.8 and
        nodo.get("estado") == "activo" and
        nodo.get("d2EPI_dt2", 0) < -0.05
    )

    if not condiciones:
        return False

    # Aplicar contracción resonante
    nodo["EPI"] = round(nodo["EPI"] * 0.7, 3)
    nodo["estado"] = "latente"
    nodo["glifo"] = "NUL"
    nodo["categoria"] = "contractivo"

    historial_glifos_por_nodo.setdefault(nodo_id, []).append((paso, "NUL"))
    
    return True

def activar_val_si_estabilidad(nodo_id, G, paso, historial_glifos_por_nodo):
    nodo = G.nodes[nodo_id]

    # Restricción por sobreexpansión
    if nodo.get("expansiones_val", 0) >= 3:
        return None

    condiciones = (
        nodo.get("Si", 0) > 0.85 and
        abs(nodo.get("ΔNFR", 0)) < 0.2 and
        nodo.get("dEPI_dt", 0) > 0.18 and
        nodo.get("d2EPI_dt2", 0) > 0.2 and
        nodo.get("estado") == "activo"
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
        "historial_Si": [(paso, nodo["Si"])]
    }

    G.add_node(nuevo_id, **nuevo_nodo)
    G.add_edge(nodo_id, nuevo_id)

    historial_glifos_por_nodo.setdefault(nodo_id, []).append((paso, "VAL"))
    historial_glifos_por_nodo[nuevo_id] = [(paso, "VAL")]

    nodo["expansiones_val"] = nodo.get("expansiones_val", 0) + 1

    return nuevo_id

def aplicar_remesh_grupal(G, historial_glifos_por_nodo):
    nodos_aplicados = set()

    for nodo_id in G.nodes:
        if nodo_id in nodos_aplicados:
            continue

        historial = historial_glifos_por_nodo.get(nodo_id, [])
        if len(historial) < 3:
            continue

        ultimos_glifos = [g for _, g in historial[-3:]]
        if len(set(ultimos_glifos)) != 1:
            continue

        glifo_recurrente = ultimos_glifos[0]

        vecinos = list(G.neighbors(nodo_id))
        grupo = [nodo_id]

        for v_id in vecinos:
            v_nodo = G.nodes[v_id]
            v_hist = historial_glifos_por_nodo.get(v_id, [])
            if len(v_hist) >= 3:
                if [g for _, g in v_hist[-3:]] == ultimos_glifos:
                    if abs(v_nodo.get("θ", 0) - G.nodes[nodo_id].get("θ", 0)) < 0.1:
                        if abs(v_nodo.get("EPI", 0) - v_nodo.get("EPI_prev", v_nodo.get("EPI", 0))) < 0.01:
                            if v_nodo.get("ΔNFR", 1.0) < 0.2:
                                grupo.append(v_id)

        if len(grupo) >= 3:
            for g_id in grupo:
                g_nodo = G.nodes[g_id]
                g_nodo["EPI_prev"] = g_nodo.get("EPI_prev", g_nodo["EPI"])
                g_nodo["EPI_prev2"] = g_nodo.get("EPI_prev2", g_nodo["EPI"])
                g_nodo["EPI"] = (g_nodo["EPI_prev"] + g_nodo["EPI_prev2"]) / 2
                g_nodo["Si"] *= 0.98
                g_nodo["νf"] *= 0.98
                g_nodo["ΔNFR"] *= 0.95
                g_nodo["glifo"] = "REMESH"
                ultimo_paso = historial_glifos_por_nodo[g_id][-1][0] if historial_glifos_por_nodo[g_id] else 0
                historial_glifos_por_nodo[g_id].append((ultimo_paso + 1, "REMESH"))
                nodos_aplicados.add(g_id)

def cumple_condiciones_emergencia(forma_base, campo_coherencia):
    """
    Evalúa si una forma puede generar un NFR según criterios TNFR.
    
    Condiciones de emergencia nodal:
    1. Frecuencia estructural mínima νf > 0.3
    2. Coherencia interna suficiente (estructura no degenerada)  
    3. Acoplamiento posible con campo de coherencia
    """
    if not forma_base or len(forma_base) < 2:
        return False
    
    # Evaluar diversidad estructural interna
    diversidad = len(set(forma_base)) / len(forma_base)
    if diversidad < 0.3:  # demasiado repetitivo
        return False
    
    # Evaluar potencial de frecuencia resonante
    freq_potencial = calcular_frecuencia_resonante(forma_base)
    if freq_potencial < 0.3:  # frecuencia insuficiente para emergencia
        return False
    
    # Evaluar compatibilidad con campo de coherencia
    if campo_coherencia and len(campo_coherencia) > 0:
        coherencia_promedio = np.mean([nodo.get("EPI", 1.0) for nodo in campo_coherencia.values()])
        if coherencia_promedio > 0 and freq_potencial > coherencia_promedio * 2.5:
            return False  # demasiado energético para el campo actual
    
    return True

def evaluar_coherencia_estructural(forma_base):
    """
    Calcula EPI basado en estructura interna real según TNFR.
    
    Evalúa:
    - Simetría funcional de la forma
    - Estabilidad topológica interna  
    - Resistencia a mutaciones
    """
    if not forma_base:
        return 1.0
    
    # Análisis de simetría funcional
    forma_norm = forma_base.lower()
    longitud = len(forma_norm)
    
    # Factor de simetría: evalúa patrones internos
    def calcular_simetria(s):
        centro = len(s) // 2
        if len(s) % 2 == 0:
            izq, der = s[:centro], s[centro:][::-1]
        else:
            izq, der = s[:centro], s[centro+1:][::-1]
        
        coincidencias = sum(1 for a, b in zip(izq, der) if a == b)
        return coincidencias / max(len(izq), 1)
    
    simetria = calcular_simetria(forma_norm)
    
    # Factor de diversidad estructural
    diversidad = len(set(forma_norm)) / longitud
    
    # Factor de estabilidad (resistencia a mutaciones puntuales)
    # Basado en la distribución de caracteres
    contador = Counter(forma_norm)
    entropia = -sum((freq/longitud) * np.log2(freq/longitud) for freq in contador.values())
    estabilidad = min(1.0, entropia / 3.0)  # normalizada
    
    # Factor de coherencia por patrones vocálicos/consonánticos
    vocales = "aeiouáéíóúü"
    patron_vocal = sum(1 for c in forma_norm if c in vocales) / longitud
    coherencia_fonetica = min(1.0, abs(0.4 - patron_vocal) * 2.5)  # óptimo cerca de 40% vocales
    
    # Combinar factores según pesos TNFR
    EPI = (
        0.3 * simetria +           # simetría estructural
        0.25 * diversidad +        # diversidad interna
        0.25 * estabilidad +       # resistencia mutacional
        0.2 * coherencia_fonetica  # coherencia fónica
    )
    
    # Normalizar al rango [0.5, 2.5] típico de EPIs
    EPI_normalizada = 0.5 + EPI * 2.0
    
    return round(EPI_normalizada, 3)

def generar_matriz_coherencia(forma_base):
    """
    Crea matriz Wi(t) para evaluar estabilidad topológica interna.
    
    Modela subnodos internos como caracteres y sus acoplamientos.
    """
    if not forma_base or len(forma_base) < 2:
        return np.array([[1.0]])
    
    longitud = len(forma_base)
    Wi = np.zeros((longitud, longitud))
    
    # Acoplamiento entre caracteres adyacentes (fuerte)
    for i in range(longitud - 1):
        Wi[i][i+1] = Wi[i+1][i] = 0.8
    
    # Acoplamiento entre caracteres similares (débil)
    for i in range(longitud):
        for j in range(i+2, longitud):
            if forma_base[i].lower() == forma_base[j].lower():
                Wi[i][j] = Wi[j][i] = 0.3
    
    # Autocoherencia (diagonal)
    np.fill_diagonal(Wi, 1.0)
    
    # Normalizar filas para que sumen aproximadamente 1
    for i in range(longitud):
        suma_fila = np.sum(Wi[i])
        if suma_fila > 0:
            Wi[i] = Wi[i] / suma_fila
    
    return Wi

def sincronizar_con_campo(campo_coherencia, νf_nodo):
    """
    Calcula fase del nodo respecto al campo de coherencia global.
    
    La fase determina si el nodo está sincronizado o en disonancia
    con el estado actual de la red.
    """
    if not campo_coherencia or len(campo_coherencia) == 0:
        return 0.0  # fase neutra si no hay campo
    
    # Calcular frecuencia promedio del campo
    frecuencias_campo = [nodo.get("νf", 1.0) for nodo in campo_coherencia.values()]
    freq_promedio_campo = np.mean(frecuencias_campo)
    
    # Calcular diferencia de fase basada en frecuencias
    diferencia_freq = abs(νf_nodo - freq_promedio_campo)
    
    # Convertir a fase: diferencias pequeñas = sincronización, grandes = disonancia  
    if diferencia_freq < 0.1:
        fase = 0.0      # sincronización perfecta
    elif diferencia_freq < 0.3:
        fase = 0.25     # sincronización parcial
    elif diferencia_freq < 0.6:
        fase = 0.5      # neutral
    elif diferencia_freq < 1.0:
        fase = 0.75     # disonancia parcial
    else:
        fase = 1.0      # disonancia completa
    
    return round(fase, 3)

def gestionar_conexiones_canonico(G, paso, historia_Ct):
    """
    Reemplaza la gestión manual de conexiones por sistema canónico TNFR.
    Esta función debe reemplazar el bloque de gestión de aristas en simular_emergencia().
    """
    # Calcular coherencia total actual
    if len(G.nodes) == 0:
        C_t = 0
    else:
        C_t = sum(G.nodes[n]["EPI"] for n in G.nodes) / len(G)

    # Calcular densidad nodal promedio
    densidad_promedio = sum(len(list(G.neighbors(n))) for n in G.nodes) / len(G.nodes) if G.nodes else 0

    # Detectar fase actual de simulación
    fase_actual = detectar_fase_simulacion(G, paso, historia_Ct)

    # Calcular umbrales dinámicos
    umbrales = calcular_umbrales_dinamicos(C_t, densidad_promedio, fase_actual)

    # Aplicar gestión de conexiones canónica
    estadisticas = aplicar_umbrales_dinamicos_conexiones(G, umbrales)

    return umbrales, estadisticas

def detectar_fase_simulacion(G, paso_actual, historial_C_t, ventana=50):
    """
    Detecta la fase actual de la simulación para ajustar umbrales.

    Args:
        G: Grafo actual
        paso_actual: Paso de simulación actual
        historial_C_t: Historia de coherencia total [(paso, C_t), ...]
        ventana: Ventana de pasos para análisis de tendencias

    Returns:
        str: "emergencia", "estabilizacion", "bifurcacion"
    """
    if len(historial_C_t) < ventana:
        return "emergencia"

    # Analizar últimos valores de C(t)
    valores_recientes = [c_t for _, c_t in historial_C_t[-ventana:]]

    # Calcular variabilidad
    variabilidad = np.std(valores_recientes)
    tendencia = np.mean(valores_recientes[-10:]) - np.mean(valores_recientes[:10])

    # Contar nodos activos
    nodos_activos = sum(1 for n in G.nodes if G.nodes[n].get("estado") == "activo")
    fraccion_activa = nodos_activos / len(G.nodes) if G.nodes else 0

    # Lógica de clasificación
    if variabilidad > 0.3 and abs(tendencia) > 0.2:
        return "bifurcacion"  # alta variabilidad y cambio direccional
    elif variabilidad < 0.05 and fraccion_activa > 0.6:
        return "estabilizacion"  # baja variabilidad, muchos nodos activos
    else:
        return "emergencia"  # estado por defecto

def calcular_umbrales_dinamicos(C_t, densidad_nodal, fase_simulacion="emergencia"):

    # Factor de sensibilidad basado en desviación de C(t) del punto de equilibrio
    equilibrio_base = 1.0
    desviacion_C_t = abs(C_t - equilibrio_base)

    # Sensibilidad adaptativa: más restrictivo cuando C(t) está lejos del equilibrio
    sensibilidad = max(0.4, min(2.0, 1.0 + 0.8 * desviacion_C_t))

    # Factor de densidad: redes densas requieren umbrales más estrictos
    factor_densidad = max(0.7, min(1.5, 1.0 - 0.1 * (densidad_nodal - 3.0)))

    # Ajuste por fase de simulación
    multiplicadores_fase = {
        "emergencia": 1.2,    # más tolerante para permitir emergencia inicial
        "estabilizacion": 0.8, # más restrictivo para consolidar estructuras
        "bifurcacion": 1.5     # muy tolerante para permitir reorganización
    }

    factor_fase = multiplicadores_fase.get(fase_simulacion, 1.0)

    # Cálculo de umbrales fundamentales
    sensibilidad_final = sensibilidad * factor_densidad * factor_fase

    return {
        # Umbrales de conexión (para crear/eliminar aristas)
        'θ_conexion': 0.12 * sensibilidad_final,
        'EPI_conexion': 1.8 * sensibilidad_final,
        'νf_conexion': 0.2 * sensibilidad_final,
        'Si_conexion': 0.25 * sensibilidad_final,

        # Umbrales críticos nodales
        'θ_mutacion': 0.25 * sensibilidad_final,      # para activar Z'HIR
        'θ_colapso': 0.45 * sensibilidad_final,       # para activar SH'A
        'θ_autoorganizacion': 0.35 * sensibilidad_final, # para activar T'HOL

        # Límites de estabilidad estructural
        'EPI_max_dinamico': max(2.5, C_t * 2.8),     # límite superior adaptativo
        'EPI_min_coherencia': max(0.4, C_t * 0.3),   # límite inferior para coherencia

        # Umbrales de bifurcación estructural
        'bifurcacion_aceleracion': 0.15 * sensibilidad_final,
        'bifurcacion_gradiente': 0.8 * sensibilidad_final,

        # Metadatos de cálculo
        'C_t_usado': C_t,
        'sensibilidad_calculada': sensibilidad_final,
        'factor_densidad': factor_densidad,
        'fase': fase_simulacion
    }

def calcular_umbrales_dinamicos(C_t, densidad_nodal, fase_simulacion="emergencia"):

    # Factor de sensibilidad basado en desviación de C(t) del punto de equilibrio
    equilibrio_base = 1.0
    desviacion_C_t = abs(C_t - equilibrio_base)

    # Sensibilidad adaptativa: más restrictivo cuando C(t) está lejos del equilibrio
    sensibilidad = max(0.4, min(2.0, 1.0 + 0.8 * desviacion_C_t))

    # Factor de densidad: redes densas requieren umbrales más estrictos
    factor_densidad = max(0.7, min(1.5, 1.0 - 0.1 * (densidad_nodal - 3.0)))

    # Ajuste por fase de simulación
    multiplicadores_fase = {
        "emergencia": 1.2,    # más tolerante para permitir emergencia inicial
        "estabilizacion": 0.8, # más restrictivo para consolidar estructuras
        "bifurcacion": 1.5     # muy tolerante para permitir reorganización
    }

    factor_fase = multiplicadores_fase.get(fase_simulacion, 1.0)

    # Cálculo de umbrales fundamentales
    sensibilidad_final = sensibilidad * factor_densidad * factor_fase

    return {
        # Umbrales de conexión (para crear/eliminar aristas)
        'θ_conexion': 0.12 * sensibilidad_final,
        'EPI_conexion': 1.8 * sensibilidad_final,
        'νf_conexion': 0.2 * sensibilidad_final,
        'Si_conexion': 0.25 * sensibilidad_final,

        # Umbrales críticos nodales
        'θ_mutacion': 0.25 * sensibilidad_final,      # para activar Z'HIR
        'θ_colapso': 0.45 * sensibilidad_final,       # para activar SH'A
        'θ_autoorganizacion': 0.35 * sensibilidad_final, # para activar T'HOL

        # Límites de estabilidad estructural
        'EPI_max_dinamico': max(2.5, C_t * 2.8),     # límite superior adaptativo
        'EPI_min_coherencia': max(0.4, C_t * 0.3),   # límite inferior para coherencia

        # Umbrales de bifurcación estructural
        'bifurcacion_aceleracion': 0.15 * sensibilidad_final,
        'bifurcacion_gradiente': 0.8 * sensibilidad_final,

        # Metadatos de cálculo
        'C_t_usado': C_t,
        'sensibilidad_calculada': sensibilidad_final,
        'factor_densidad': factor_densidad,
        'fase': fase_simulacion
    }

def aplicar_umbrales_dinamicos_conexiones(G, umbrales):
    """
    Aplica umbrales dinámicos para gestión de conexiones de red.

    Args:
        G: Grafo de red
        umbrales: Umbrales calculados dinámicamente

    Returns:
        dict: Estadísticas de conexiones creadas/eliminadas
    """
    conexiones_creadas = 0
    conexiones_eliminadas = 0
    nodos_lista = list(G.nodes)

    for i in range(len(nodos_lista)):
        for j in range(i + 1, len(nodos_lista)):
            n1, n2 = nodos_lista[i], nodos_lista[j]
            nodo1, nodo2 = G.nodes[n1], G.nodes[n2]

            # Evaluar condiciones de resonancia con umbrales dinámicos
            condiciones_resonancia = [
                abs(nodo1.get("θ", 0) - nodo2.get("θ", 0)) < umbrales['θ_conexion'],
                abs(nodo1.get("EPI", 0) - nodo2.get("EPI", 0)) < umbrales['EPI_conexion'],
                abs(nodo1.get("νf", 1) - nodo2.get("νf", 1)) < umbrales['νf_conexion'],
                abs(nodo1.get("Si", 0) - nodo2.get("Si", 0)) < umbrales['Si_conexion']
            ]

            # Criterio: al menos 3 de 4 condiciones cumplidas
            resonancia_suficiente = sum(condiciones_resonancia) >= 3

            # Verificar saturación de conexiones
            vecinos_n1 = len(list(G.neighbors(n1)))
            vecinos_n2 = len(list(G.neighbors(n2)))
            max_conexiones = int(8 * umbrales['sensibilidad_calculada'])

            saturacion = vecinos_n1 >= max_conexiones and vecinos_n2 >= max_conexiones

            # Lógica de conexión/desconexión
            existe_conexion = G.has_edge(n1, n2)

            if resonancia_suficiente and not saturacion and not existe_conexion:
                G.add_edge(n1, n2)
                conexiones_creadas += 1
            elif not resonancia_suficiente and existe_conexion:
                G.remove_edge(n1, n2)
                conexiones_eliminadas += 1

    return {
        'conexiones_creadas': conexiones_creadas,
        'conexiones_eliminadas': conexiones_eliminadas,
        'umbrales_usados': umbrales
    }

__all__ = [
    'inicializar_nfr_emergente',
    '_deben_conectarse_canonico',
    'calcular_frecuencia_resonante',
    'crear_red_desde_datos', 
    'simular_emergencia',
    'aplicar_contraccion_nul',
    'activar_val_si_estabilidad',
    'aplicar_remesh_grupal',
    'cumple_condiciones_emergencia',
    'evaluar_coherencia_estructural',
    'generar_matriz_coherencia',
    'sincronizar_con_campo',
    'gestionar_conexiones_canonico',
    'calcular_umbrales_dinamicos',
    'aplicar_umbrales_dinamicos_conexiones'
]
