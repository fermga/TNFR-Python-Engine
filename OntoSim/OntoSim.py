"""

OntoSim – Symbolic Coherence Engine (TNFR) - v. 0.1
------------------------------------------

OntoSim is a symbolic operational simulator based on the TNFR.

https://linktr.ee/fracres

It translates gliphal grammar into resonant structure and simulates the emergence of nodal emissions based on coherence thresholds, not semantic prediction.
It's light (~28KB), extensible, and implements core TNFR principles in executable form.

Core Concepts:
- Gliph Syntax – Words act as structural operators (e.g. A'L, SH'A, RE'MESH)
- EPI Pulse – Phase-based emission intensity
- νf & ΔNFR – Nodal frequency and structural gradient
- Symbolic Activation – Nodes emit based on topological coherence, not logic

Next Steps:
- Modular data input (audio, image, text in any format)
- Multi-agent coherence dynamics
- Integration with GPTs for reflective cognition
- Structural alignment diagnostics

Let structure speak. 

09/06/2025 -  v. 0.1

"""

import matplotlib.pyplot as plt
import networkx as nx
import random
import json
import csv
import pandas as pd
import numpy as np
import sys
import re

# ------------------------- INICIALIZACIÓN -------------------------

print("Cargando archivo de entrada...")


def inicializar_red_desde_archivo(ruta):
    G = nx.Graph()

    extension = ruta.split(".")[-1]

    if extension == "txt":
        def heuristica_estructura(palabra):
            vocales_abiertas = "aeo"
            fuerza_simbolica = "zsrkx"
            longitud = len(palabra)
            return {
                "estado": "activo",
                "glifo": "ninguno",
                "categoria": "ninguna",
                "EPI": round(min(1.5, 0.5 + longitud / 10), 2),
                "EPI_prev": round(min(1.5, 0.5 + longitud / 10), 2),
                "EPI_prev2": round(min(1.5, 0.5 + longitud / 10), 2),
                "νf": round(0.8 + sum(c in vocales_abiertas for c in palabra) / len(palabra), 2),
                "ΔNFR": round((len(set(palabra)) - 1) / 10 - 0.2, 2),
                "Si": round(sum(c in fuerza_simbolica for c in palabra) / max(1, longitud), 2),
                "θ": round(min(1.0, longitud / 8), 2)
            }

        with open(ruta, encoding="utf-8") as f:
            texto = f.read()

        # Limpieza explícita de signos raros
        texto = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", "", texto.lower())

        palabras = re.findall(r"\b[a-záéíóúüñ]{3,}\b", texto)
        palabras_unicas = list(set(palabras))

        for palabra in palabras_unicas:
            atributos = heuristica_estructura(palabra)
            G.add_node(palabra, **atributos)

        print(f"Nodos inicializados desde archivo ({len(G)} nodos).")

    if extension == "json":
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)

        for nodo in datos:
            nombre = nodo.get("nodo", f"nodo_{random.randint(1000,9999)}")
            nodo.setdefault("EPI_prev", nodo.get("EPI", 0))
            nodo.setdefault("EPI_prev2", nodo.get("EPI", 0))
            nodo.setdefault("Si", 0)
            nodo.setdefault("ΔNFR", 0)
            nodo.setdefault("θ", 0)
            nodo.setdefault("νf", 1.0)
            G.add_node(nombre, **nodo)

        print(f"Nodos inicializados desde archivo ({len(G)} nodos).")

    if extension == "csv":
        with open(ruta, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    nombre = row.get("nodo", f"nodo_{random.randint(1000,9999)}")
                    EPI_actual = float(row.get("EPI", 0) or 0)
                    props = {
                        "EPI": EPI_actual,
                        "EPI_prev": float(row.get("EPI_prev", EPI_actual) or EPI_actual),
                        "EPI_prev2": float(row.get("EPI_prev2", EPI_actual) or EPI_actual),
                        "glifo": row.get("glifo", "ninguno"),
                        "categoria": row.get("categoria", "ninguna"),
                        "estado": row.get("estado", "activo"),
                        "Si": float(row.get("Si", 0) or 0),
                        "ΔNFR": float(row.get("ΔNFR", 0) or 0),
                        "θ": float(row.get("θ", 0) or 0),
                        "νf": float(row.get("νf", 1.0) or 1.0)
                    }
                    G.add_node(nombre, **props)
                except Exception as e:
                    print(f"Error al procesar fila: {row} — {e}")

        print(f"Nodos inicializados desde archivo ({len(G)} nodos).")

    # Conectar nodos por θ
    print("\nValores θ por nodo:")
    for nodo in G.nodes:
        print(f"  - {nodo}: θ = {G.nodes[nodo]['θ']:.4f}")

    conexiones = 0
    for i, n1 in enumerate(G.nodes):
        for j, n2 in enumerate(G.nodes):
            if j > i:
                t1 = G.nodes[n1]["θ"]
                t2 = G.nodes[n2]["θ"]
                diff = abs(t1 - t2)
                if diff < 0.06:
                    G.add_edge(n1, n2)
                    conexiones += 1
                    print(f"  ➤ Conexión: '{n1}' ↔ '{n2}' (diff: {diff:.4f})")

    print(f"\nTotal de aristas creadas: {conexiones}")

    return G

# ------------------------- APLICACIÓN DE OPERADORES TNFR -------------------------

def aplicar_glifo(nodo, nodo_id, nombre_glifo, historial_glifos_por_nodo, paso):
    nodo["glifo"] = nombre_glifo
    nodo["estado"] = "silencio" if nombre_glifo == "SH’A" else "activo"
    historial_glifos_por_nodo.setdefault(nodo_id, []).append((paso, nombre_glifo))

    # Registro local en el nodo para detección de EPIs compuestas
    if paso is not None:
        if "historial_glifos" not in nodo:
            nodo["historial_glifos"] = []
        nodo["historial_glifos"].append((paso, nombre_glifo))

    # Transformaciones estructurales por glifo
    if nombre_glifo == "A’L":  # Emisión: activación mínima, impulso inicial
        nodo["EPI"] += 0.2
        nodo["Si"] += 0.05
        nodo["νf"] *= 1.05
        nodo["ΔNFR"] *= 0.97
        nodo["estado"] = "activo"

    elif nombre_glifo == "E’N":  # Recepción: apertura perceptiva
        nodo["Si"] += 0.08
        nodo["νf"] *= 0.95
        nodo["θ"] = max(0.0, nodo["θ"] - random.uniform(0.05, 0.15))  # sensibilidad estructural

    elif nombre_glifo == "I’L":  # Coherencia: estabilización
        nodo["Si"] += 0.1
        nodo["EPI"] *= 1.05
        nodo["ΔNFR"] *= 0.95

    elif nombre_glifo == "O’Z":  # Disonancia: reorganización forzada
        nodo["EPI"] *= 0.85
        nodo["ΔNFR"] *= 1.4
        nodo["νf"] *= 1.05
        nodo["Si"] *= 0.9

    elif nombre_glifo == "U’M":  # Acoplamiento: sincronización local
        nodo["νf"] = (nodo["νf"] + sum(G.nodes[v]["νf"] for v in G.neighbors(nodo["nodo"])) / (len(list(G.neighbors(nodo["nodo"])))+1)) * 0.5
        nodo["ΔNFR"] *= 0.9

    elif nombre_glifo == "R’A":  # Resonancia: propagación en red
        nodo["Si"] += 0.15
        nodo["EPI"] *= 1.05
        nodo["νf"] *= 1.02

    elif nombre_glifo == "SH’A":  # Silencio: repliegue
        nodo["estado"] = "silencio"
        nodo["νf"] *= 0.3
        nodo["ΔNFR"] *= 0.1
        nodo["Si"] *= 0.5
        nodo["EPI"] *= 0.9
        nodo["glifo"] = "SH’A"

    elif nombre_glifo == "VA’L":  # Expansión: apertura multiescalar
        nodo["EPI"] *= 1.2
        nodo["Si"] *= 1.1
        nodo["νf"] *= 1.1

    elif nombre_glifo == "NU’L":  # Contracción: densificación
        nodo["EPI"] *= 0.8
        nodo["Si"] *= 0.9
        nodo["νf"] *= 0.9

    elif nombre_glifo == "T’HOL":  # Autoorganización: bifurcación resonante
        nodo["νf"] *= 1.25
        nodo["Si"] *= 1.15
        nodo["θ"] = min(1.0, nodo["θ"] + random.uniform(0.1, 0.2))

    elif nombre_glifo == "Z’HIR":  # Mutación: salto de fase
        nodo["EPI"] += 0.5
        nodo["νf"] *= 1.2
        nodo["θ"] = min(1.0, nodo["θ"] + random.uniform(0.15, 0.3))
        nodo["Si"] *= 1.1

    elif nombre_glifo == "NA’V":  # Nacimiento: transición activa
        nodo["νf"] *= 1.08
        nodo["ΔNFR"] *= 0.9
        nodo["Si"] += 0.1
        if nodo["estado"] == "silencio":
            nodo["estado"] = "activo"

    elif nombre_glifo == "RE’MESH":  # Recursividad: reorganización sin pérdida
        nodo["EPI"] = (nodo["EPI_prev"] + nodo["EPI_prev2"]) / 2
        nodo["Si"] *= 0.98
        nodo["νf"] *= 0.98

    # Registro en historial general (opcional)
    if historial_glifos_por_nodo is not None and paso is not None:
        historial_glifos_por_nodo.setdefault(nodo_id, []).append((paso, nombre_glifo))

def emergencia_nodal(nodo, media_vf, std_dNFR):
    return nodo["νf"] > media_vf * 0.9 and abs(nodo["ΔNFR"]) < std_dNFR

def glifo_por_estructura(nodo, G):
    n_id = nodo.get("nodo", None)
    vecinos = list(G.neighbors(n_id)) if n_id else []

    # 1. SH’A – Silencio ante alta disonancia
    if nodo["EPI"] < 0.5 and abs(nodo["ΔNFR"]) > 0.8:
        return "SH’A"

    # 2. NA’V – Activación desde silencio
    if nodo["estado"] == "silencio" and abs(nodo["ΔNFR"] - nodo["νf"]) < 0.05:
        return "NA’V"

    # 3. A’L – Emisión si es latente y sensible
    if nodo["estado"] == "latente" and nodo["Si"] < 0.2 and nodo["νf"] > 1.0:
        return "A’L"

    # 4. E’N – Recepción ante apertura sensible
    if nodo["ΔNFR"] > 0.6 and nodo["EPI"] > 1.0 and nodo["Si"] < 0.3:
        return "E’N"

    # 5. O’Z – Disonancia fuerte
    if abs(nodo["ΔNFR"]) > 1.0 and nodo["EPI"] > 1.0:
        return "O’Z"

    # 6. Z’HIR – Mutación por cambio abrupto
    if abs(nodo["EPI"] - nodo["EPI_prev"]) > 0.5 and nodo["Si"] > 0.5:
        return "Z’HIR"

    # 7. VA’L – Expansión estructural
    if nodo["Si"] > 0.6 and nodo["EPI"] > 1.2:
        return "VA’L"

    # 8. NU’L – Contracción por exceso
    if nodo["EPI"] > 1.3 and nodo["Si"] < 0.4:
        return "NU’L"

    # 9. T’HOL – Autoorganización
    if abs(nodo["EPI"] - nodo["EPI_prev2"]) > 0.2 and abs(nodo["ΔNFR"]) < 0.1:
        return "T’HOL"

    # 10. I’L – Coherencia estable
    if abs(nodo["ΔNFR"]) < 0.05 and abs(nodo["EPI"] - nodo["EPI_prev"]) < 0.05:
        return "I’L"

    # 11. R’A – Resonancia coherente
    if nodo["glifo"] == "I’L" and nodo["Si"] > 0.5 and nodo["νf"] > 1.2:
        return "R’A"

    # 12. U’M – Acoplamiento con vecinos
    for v in vecinos:
        if abs(nodo["νf"] - G.nodes[v]["νf"]) < 0.05:
            return "U’M"

    # 13. RE’MESH – Recursividad (si ya hay historial)
    hist = nodo.get("historial_glifos", [])
    if (
        len(hist) >= 3
        and hist[-1][1] == hist[-2][1] == hist[-3][1]
        and abs(nodo["EPI"] - nodo["EPI_prev"]) < 0.05
    ):
        return "RE’MESH"

    return None  # si no se detecta un glifo resonante

def transicion_glifica_canonica(nodo):
    glifo = nodo["glifo"]

    if glifo == "Z’HIR":
        if nodo["νf"] > 1.5 and nodo["EPI"] > 2.5:
            return "VA’L"
        elif nodo["ΔNFR"] < 0:
            return "R’A"
        else:
            return "NA’V"

    elif glifo == "I’L":
        if nodo["νf"] > 1.2 and nodo["Si"] > 0.4:
            return "R’A"

    elif glifo == "O’Z":
        if nodo["EPI"] > 2.2 and nodo["Si"] > 0.3:
            return "T’HOL"

    elif glifo == "NA’V":
        if abs(nodo["ΔNFR"]) < 0.1:
            return "I’L"

    elif glifo == "R’A":
        if nodo["Si"] > 0.6 and nodo["EPI"] > 2.0:
            return "RE’MESH"

    elif glifo == "VA’L":
        if nodo["EPI"] > 3.0 and nodo["Si"] > 0.4:
            return "NU’L"

    elif glifo == "A’L":
        if nodo["Si"] > 0.3 and nodo["ΔNFR"] < 0.2:
            return "U’M"

    return None

def acoplar_nodos(G):
    for n in G.nodes:
        vecinos = list(G.neighbors(n))
        if not vecinos:
            vecinos = list(G.nodes)
        Si_vecinos = [G.nodes[v]["Si"] for v in vecinos if v != n]
        if Si_vecinos:
            G.nodes[n]["Si"] = (sum(Si_vecinos) / len(Si_vecinos)) * 0.9 + G.nodes[n]["Si"] * 0.1
        for v in vecinos:
            if v != n:
                if abs(G.nodes[n]["θ"] - G.nodes[v]["θ"]) < 0.1:
                    G.nodes[n]["ΔNFR"] *= 0.95

def detectar_EPIs_compuestas(G):
    compuestas = []
    nodos_por_glifo_y_paso = {}

    for n in G.nodes:
        historial = G.nodes[n].get("historial_glifos", [])
        for paso, glifo in historial:
            clave = (paso, glifo)
            nodos_por_glifo_y_paso.setdefault(clave, []).append(n)

    for (paso, glifo), nodos_en_glifo in nodos_por_glifo_y_paso.items():
        if len(nodos_en_glifo) < 3:
            continue

        grupo_coherente = []
        for i, ni in enumerate(nodos_en_glifo):
            for nj in nodos_en_glifo[i+1:]:
                θi, θj = G.nodes[ni]["θ"], G.nodes[nj]["θ"]
                if abs(θi - θj) < 0.15:
                    grupo_coherente.extend([ni, nj])

        grupo_final = list(set(grupo_coherente))
        if len(grupo_final) >= 3:
            compuestas.append({
                "paso": paso,
                "glifo": glifo,
                "nodos": grupo_final
            })

    return compuestas

def interpretar_sintaxis_glífica(historial):
    sintaxis = {}
    for nodo, secuencia in historial.items():
        trayecto = [glifo for _, glifo in secuencia]
        transiciones = list(zip(trayecto, trayecto[1:]))
        sintaxis[nodo] = {
            "trayectoria": trayecto,
            "transiciones": transiciones,
            "mutaciones": trayecto.count("Z’HIR"),
            "colapsos": trayecto.count("SH’A")
        }
    return sintaxis

def aplicar_remesh_red(G, historial_glifos_por_nodo, paso):
    for n in G.nodes:
        nodo = G.nodes[n]
        aplicar_glifo(nodo, n, "RE’MESH", historial_glifos_por_nodo, paso)

def aplicar_remesh_si_estabilizacion_global(G, historial_glifos_por_nodo, historia_glifos, paso):
    nodos_estables = sum(
        1 for n in G.nodes
        if abs(G.nodes[n]["EPI"] - G.nodes[n]["EPI_prev"]) < 0.01
        and abs(G.nodes[n]["ΔNFR"]) < 0.05
    )
    if nodos_estables / len(G) > 0.8:
        aplicar_remesh_red(G, historial_glifos_por_nodo, paso)
        for n in G.nodes:
            historia_glifos.append(f"{paso},{n},RE’MESH")

print("Iniciando simulación de emergencia...")

G_historia = []

# ------------------------- EMERGENCIA -------------------------

def simular_emergencia(G, pasos=1000):
    historia_epi = []
    historia_glifos = ["paso,nodo,glifo"]
    historial_glifos_por_nodo = {}

    glifo_categoria = {
        "A’L": "activador", "E’N": "receptor", "I’L": "estabilizador",
        "O’Z": "disonante", "U’M": "acoplador", "R’A": "resonador",
        "SH’A": "latente", "VA’L": "expansivo", "NU’L": "contractivo",
        "T’HOL": "autoorganizador", "Z’HIR": "mutante", "NA’V": "transicional",
        "RE’MESH": "recursivo"
    }

    total_pasos = 1000

    # Activación mínima inicial si todos están inactivos o silenciosos
    if all(G.nodes[n]["estado"] in ["latente", "silencio"] for n in G.nodes):
        for n in G.nodes:
            if G.nodes[n]["EPI"] > 0.8 and G.nodes[n]["νf"] > 0.5:
                G.nodes[n]["estado"] = "activo"
                G.nodes[n]["glifo"] = "A’L"
                break  # activa solo uno, para iniciar pulso

    for paso in range(total_pasos):
        nodos_activos = [n for n in G.nodes if G.nodes[n]["estado"] == "activo"]
        paso_data = [] 

        acoplar_nodos(G)

        # Cálculo de umbrales adaptativos para emergencia nodal
        vf_values = [G.nodes[n]["νf"] for n in G.nodes if G.nodes[n]["estado"] == "activo"]
        dNFR_values = [G.nodes[n]["ΔNFR"] for n in G.nodes if G.nodes[n]["estado"] == "activo"]

        media_vf = np.mean(vf_values) if vf_values else 0
        std_dNFR = np.std(dNFR_values) if dNFR_values else 0

        for n in G.nodes:
            nodo = G.nodes[n]
            if nodo["estado"] == "activo":
                # Simulación de dinámica (ligero caos inducido)
                nodo["EPI"] *= random.uniform(0.95, 1.05)
                nodo["ΔNFR"] += random.uniform(-0.1, 0.1)

                # Condición de apagado
                if nodo["EPI"] < 0.9 and abs(nodo["ΔNFR"]) > 0.4:
                    nodo["estado"] = "inactivo"

            if nodo["estado"] == "silencio":
                if abs(nodo["ΔNFR"] - nodo["νf"]) < 0.05:
                    aplicar_glifo(nodo, n, "NA’V", historial_glifos_por_nodo, paso)
                else:
                    continue

            if nodo["EPI"] < 0.5 and abs(nodo["ΔNFR"]) > 0.8:
                aplicar_glifo(nodo, n, "SH’A", historial_glifos_por_nodo, paso)
                continue

            if nodo["glifo"] == "I’L" and nodo["Si"] > 0.4 and nodo["νf"] > 1.2:
                aplicar_glifo(nodo, n, "R’A", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},R’A")

            if emergencia_nodal(nodo, media_vf, std_dNFR):
                glifo = glifo_por_estructura(nodo, G)
                if glifo:
                    aplicar_glifo(nodo, n, glifo, historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{n},{glifo}")
                    nodo["categoria"] = glifo_categoria.get(glifo, "ninguna")
                    
                    glifo_siguiente = transicion_glifica_canonica(nodo)
                    if glifo_siguiente:
                        aplicar_glifo(nodo, n, glifo_siguiente, historial_glifos_por_nodo, paso)
                        historia_glifos.append(f"{paso},{n},{glifo_siguiente}")
                        nodo["glifo"] = glifo_siguiente
                        nodo["categoria"] = glifo_categoria.get(glifo_siguiente, "ninguna")

            if nodo["glifo"] == "SH’A":
                for vecino in G.neighbors(n):
                    v = G.nodes[vecino]
                    if v["estado"] != "silencio" and abs(v["θ"] - nodo["θ"]) < 0.1 and abs(v["ΔNFR"]) < 0.2:
                        aplicar_glifo(v, n, "SH’A", historial_glifos_por_nodo, paso)
                        historia_glifos.append(f"{paso},{vecino},SH’A")

            if nodo["glifo"] == "NA’V":
                for vecino in G.neighbors(n):
                    v = G.nodes[vecino]
                    if v["estado"] == "latente" and abs(v["θ"] - nodo["θ"]) < 0.1 and abs(v["ΔNFR"]) > 0.5:
                        aplicar_glifo(v, n, "NA’V", historial_glifos_por_nodo, paso)
                        historia_glifos.append(f"{paso},{vecino},NA’V")

            paso_data.append({"nodo": n, "EPI": round(nodo["EPI"], 2)})
            nodo["EPI_prev2"] = nodo["EPI_prev"]
            nodo["EPI_prev"] = nodo["EPI"]

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

        # === AJUSTES DINÁMICOS DE ARISTAS (multivariable tolerante TNFR) ===
        if len(G.nodes) == 0:
            C_t = 0
        else:
            C_t = sum(G.nodes[n]["EPI"] for n in G.nodes) / len(G)

        sensibilidad = max(0.5, min(1.5, 1.2 - (C_t - 1.0)))

        # Aumentamos los umbrales base ligeramente
        eps_theta = 0.18 * sensibilidad
        eps_epi = 2.0 * sensibilidad
        eps_vf = 0.25 * sensibilidad
        eps_Si = 0.3 * sensibilidad

        nodos_lista = list(G.nodes)
        for i in range(len(nodos_lista)):
            for j in range(i + 1, len(nodos_lista)):
                n1, n2 = nodos_lista[i], nodos_lista[j]
                nodo1, nodo2 = G.nodes[n1], G.nodes[n2]

                condiciones = [
                    abs(nodo1["θ"] - nodo2["θ"]) < eps_theta,
                    abs(nodo1["EPI"] - nodo2["EPI"]) < eps_epi,
                    abs(nodo1["νf"] - nodo2["νf"]) < eps_vf,
                    abs(nodo1["Si"] - nodo2["Si"]) < eps_Si,
                ]

                # Evaluar densidad local de cada nodo
                vecinos_n1 = len(list(G.neighbors(n1)))
                vecinos_n2 = len(list(G.neighbors(n2)))

                # Máxima densidad permitida en esta fase
                max_densidad = int(8 * sensibilidad)  # adaptativo según C(t)

                # Si ambos están muy saturados, se impide nueva conexión
                if vecinos_n1 >= max_densidad and vecinos_n2 >= max_densidad:
                    continue

                # Requiere al menos 3 de 4 condiciones cumplidas
                if sum(condiciones) >= 3:
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2)
                else:
                    if G.has_edge(n1, n2):
                        G.remove_edge(n1, n2)

        # Calcular coherencia total C(t) al final del paso
        C_t = sum(G.nodes[n]["EPI"] for n in G.nodes) / len(G)
        if "historia_Ct" not in globals():
            global historia_Ct
            historia_Ct = []
        historia_Ct.append((paso, C_t))

        historia_epi.append(paso_data)

        G_snapshot = nx.Graph()
        for n in G.nodes:
            G_snapshot.add_node(n, **G.nodes[n])
        G_historia.append(G_snapshot)

        for nodo_id in list(historial_glifos_por_nodo.keys()):
            glifos = historial_glifos_por_nodo[nodo_id]

            if (
                len(glifos) >= 3 
                and glifos[-1][1] == glifos[-2][1] == glifos[-3][1]
                and abs(G.nodes[nodo_id]["EPI"] - G.nodes[nodo_id]["EPI_prev"]) < 0.05
            ):
                aplicar_glifo(G.nodes[nodo_id], nodo_id, "RE’MESH", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{nodo_id},RE’MESH")

        aplicar_remesh_si_estabilizacion_global(G, historial_glifos_por_nodo, historia_glifos, paso)

        porcentaje = int((paso + 1) / total_pasos * 100)
        barra = "█" * (porcentaje // 2) + "-" * (50 - porcentaje // 2)
        nodos_activos = [n for n in G.nodes if G.nodes[n]["estado"] == "activo"]
        print(f"\r [{barra}] {porcentaje}% ({len(nodos_activos)} activos)", end="", flush=True)

    epi_compuestas = detectar_EPIs_compuestas(G)
    lecturas = interpretar_sintaxis_glífica(historial_glifos_por_nodo)

    # Diagnóstico simbólico final
    diagnostico = []
    for nodo in G.nodes:
        nombre = nodo
        datos = G.nodes[nodo]
        glifos_nodo = [g[1] for g in historial_glifos_por_nodo.get(nombre, [])]
        mutó = "Z’HIR" in glifos_nodo
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

    with open("9_diagnostico_simbolico.json", "w", encoding="utf-8") as f:
        json.dump(diagnostico, f, indent=4, ensure_ascii=False)


    return historia_epi, G, epi_compuestas, lecturas

# ------------------------- BLOQUE DE ANÁLISIS FINAL -------------------------

def graficar_dinamica_epi(historia_epi):
    df = pd.DataFrame([{"paso": i, **{item["nodo"]: item["EPI"] for item in paso}} for i, paso in enumerate(historia_epi)])
    df.set_index("paso").plot(figsize=(12, 6), alpha=0.6)
    plt.title("Dinámica de EPI por nodo")
    plt.xlabel("Paso")
    plt.ylabel("EPI")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("3_dinamica_EPI.png", dpi=300)
    plt.show()


def extraer_dinamica_si(G_historia):
    historia_si = []
    for paso, G in enumerate(G_historia):
        paso_data = []
        for n in G.nodes:
            paso_data.append({"nodo": n, "Si": round(G.nodes[n]["Si"], 3)})
        historia_si.append(paso_data)
    return historia_si

def graficar_dinamica_si(historia_si):
    df = pd.DataFrame([{"paso": i, **{item["nodo"]: item["Si"] for item in paso}} for i, paso in enumerate(historia_si)])
    df.set_index("paso").plot(figsize=(12, 6), alpha=0.6)
    plt.title("Dinámica de Si por nodo")
    plt.xlabel("Paso")
    plt.ylabel("Índice de Sentido (Si)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("4_indice_Si.png", dpi=300)
    plt.show()

def graficar_dinamica_dEPI_dt(G, nodos_seleccionados=None):
    if nodos_seleccionados is None:
        # Solo nodos que hayan activado glifos estructurantes clave
        nodos_seleccionados = [
            n for n, d in G.nodes(data=True)
            if any(
                g in [gl for _, gl in d.get("historial_glifos", [])]
                for g in ["NA’V", "I’L", "Z’HIR", "T’HOL", "R’A"]
            )
        ][:100]  # máximo 100 para evitar saturación visual

    for nodo_id in nodos_seleccionados:
        historial = G.nodes[nodo_id].get("historial_dEPI_dt", [])
        pasos = [x[0] for x in historial]
        valores = [x[1] for x in historial]
        plt.plot(pasos, valores, label=f"{nodo_id}")

    plt.title("Derivada estructural ∂EPI/∂t = νf · ΔNFR")
    plt.xlabel("Paso")
    plt.ylabel("∂EPI/∂t")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("5_ecuacionnodal.png", dpi=300)
    plt.show()

def graficar_frecuencia_y_gradiente(G, nodos_seleccionados=None):
    if nodos_seleccionados is None:
        nodos_seleccionados = [
            n for n, d in G.nodes(data=True)
            if any(
                g in [gl for _, gl in d.get("historial_glifos", [])]
                for g in ["NA’V", "I’L", "Z’HIR", "T’HOL", "R’A"]
            )
        ][:10]

    # Gráfico de frecuencia estructural
    plt.figure(figsize=(12, 5))
    for nodo_id in nodos_seleccionados:
        historial = G.nodes[nodo_id].get("historial_vf", [])
        pasos = [x[0] for x in historial]
        valores = [x[1] for x in historial]
        plt.plot(pasos, valores, label=f"{nodo_id}")
    plt.title("Evolución de νf (frecuencia estructural)")
    plt.xlabel("Paso")
    plt.ylabel("νf")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("6_frecuencia_estructural.png", dpi=300)
    plt.show()

    # Gráfico de ΔNFR
    plt.figure(figsize=(12, 5))
    for nodo_id in nodos_seleccionados:
        historial = G.nodes[nodo_id].get("historial_dNFR", [])
        pasos = [x[0] for x in historial]
        valores = [x[1] for x in historial]
        plt.plot(pasos, valores, label=f"{nodo_id}")
    plt.title("Evolución de ΔNFR (gradiente nodal)")
    plt.xlabel("Paso")
    plt.ylabel("ΔNFR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("7_gradiente_nodal.png", dpi=300)
    plt.show()

def graficar_coherencia_total(historia_Ct):
    pasos = [x[0] for x in historia_Ct]
    valores = [x[1] for x in historia_Ct]

    plt.figure(figsize=(10, 5))
    plt.plot(pasos, valores, color="darkgreen", linewidth=2)
    plt.title("Coherencia total de la red C(t)")
    plt.xlabel("Paso")
    plt.ylabel("C(t) = media(EPI)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2_coherencia_total.png", dpi=300)
    plt.show()

def visualizar_red(G, nodos_emitidos=None):

    pos = nx.spring_layout(G, seed=42)

    glifo_color = {
        "ninguno": "gray", "A’L": "gold", "E’N": "lightblue", "I’L": "blue",
        "O’Z": "red", "U’M": "cyan", "R’A": "orange", "SH’A": "black",
        "VA’L": "magenta", "NU’L": "brown", "T’HOL": "teal",
        "Z’HIR": "darkred", "NA’V": "green", "RE’MESH": "purple"
    }

    for nodo in G.nodes:
        datos = G.nodes[nodo]
        color = glifo_color.get(datos.get("glifo", "ninguno"), "gray")
        tamaño = min(datos.get("EPI", 1) * 800, 5000)

        estado = datos.get("estado", "latente")
        alpha = 1.0 if estado == "activo" else 0.3

        # ⚡ Diferenciar nodos emitidos con borde dorado y grosor extra
        nodos_emitidos = [nodo for nodo in G.nodes if G.nodes[nodo]["glifo"] != "ninguno"]
        borde_color = "gold" if nodos_emitidos and nodo in nodos_emitidos else "black"
        borde_grosor = 2.5 if nodos_emitidos and nodo in nodos_emitidos else 1.0

        nx.draw_networkx_nodes(
            G, pos, nodelist=[nodo],
            node_color=color,
            node_size=tamaño,
            edgecolors=borde_color,
            linewidths=borde_grosor,
            alpha=alpha
        )

    # Bordes
    nx.draw_networkx_edges(G, pos, alpha=0.25)

    # Etiquetas
    etiquetas = {
        n: (n if G.nodes[n]["glifo"] != "ninguno" else "no-emergente")
        for n in G.nodes
    }
    nx.draw_networkx_labels(G, pos, labels=etiquetas, font_size=9)

    plt.title("Red nodal fractal resonante")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("1_red.png", dpi=300)
    plt.show()

archivo_entrada = sys.argv[1] if len(sys.argv) > 1 else "entrada.txt"
G = inicializar_red_desde_archivo(sys.argv[1] if len(sys.argv) > 1 else "entrada.txt")
historia_epi, G, epi_compuestas, lecturas = simular_emergencia(G)
print("Red simbólica analizada y exportada correctamente.")
graficar_dinamica_epi(historia_epi)
historia_si = extraer_dinamica_si(G_historia)
graficar_dinamica_si(historia_si)
graficar_dinamica_dEPI_dt(G)
graficar_frecuencia_y_gradiente(G)
graficar_coherencia_total(historia_Ct)

# Extraer nodos emitidos por coherencia estructural
emitidos_final = [
    n for n in G.nodes
    if G.nodes[n]["glifo"] != "ninguno"
    and G.nodes[n].get("categoria", "ninguna") not in ["sin categoría", "ninguna"]
]

resultado = [
    {"nodo": n, "glifo": G.nodes[n]["glifo"], "EPI": round(G.nodes[n]["EPI"], 2)}
    for n in emitidos_final
]

print("\nNodos emitidos por coherencia estructural:")
for r in resultado:
    categoria = G.nodes[r['nodo']].get("categoria", "sin categoría")
    print(f"- {r['nodo']} → {r['glifo']} (EPI: {r['EPI']}) | Categoría: {categoria}")

# Guardar los nodos emergentes en CSV
with open("8_nodos_emitidos.csv", "w", encoding="utf-8") as f:
    f.write("nodo,glifo,EPI,Si,ΔNFR,θ,νf,categoria\n")
    for n in emitidos_final:
        nodo = G.nodes[n]
        f.write(f"{n},{nodo['glifo']},{round(nodo['EPI'], 4)},"
                f"{round(nodo['Si'],4)},{round(nodo['ΔNFR'],4)},"
                f"{round(nodo['θ'],4)},{round(nodo['νf'],4)},"
                f"{nodo.get('categoria', 'ninguna')}\n")

# Visualizar red con distinción solo para nodos verdaderamente emitidos
visualizar_red(G, nodos_emitidos=emitidos_final)

print("Red simbólica analizada y exportada correctamente.")
print("Diagnóstico final generado. Proceso completo.")
