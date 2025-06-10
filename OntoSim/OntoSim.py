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
from tqdm import tqdm
from math import isnan
import matplotlib.pyplot as plt
import networkx as nx
import random
import json
import csv
import pandas as pd
import numpy as np
import sys
import re
import os
import imageio.v2 as imageio
from collections import defaultdict, Counter

# ------------------------- INICIALIZACIÓN -------------------------

print("Cargando archivo de entrada...")

glifo_categoria = {
    "A’L": "emisión",
    "E’N": "recepción",
    "I’L": "coherencia",
    "O’Z": "disonancia",
    "U’M": "acoplamiento",
    "R’A": "resonancia",
    "SH’A": "silencio",
    "VA’L": "expansión",
    "NU’L": "contracción",
    "T’HOL": "autoorganización",
    "Z’HIR": "mutación",
    "NA’V": "nacimiento",
    "RE’MESH": "recursividad"
}

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
                "EPI_prev3": round(min(1.5, 0.5 + longitud / 10), 2),
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
            nodo.setdefault("EPI_prev3", nodo.get("EPI", 0))
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
                        "EPI_prev3": float(row.get("EPI_prev3", EPI_actual) or EPI_actual),
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

def normalizar_historial_glifos(historial_glifos_por_nodo, analizar_dinamica=False, expandido=False):
    glifo_codigo = {
        "A’L": 1, "E’N": 2, "I’L": 3, "O’Z": 4, "U’M": 5,
        "R’A": 6, "SH’A": 7, "VA’L": 8, "NU’L": 9, "T’HOL": 10,
        "Z’HIR": 11, "NA’V": 12, "RE’MESH": 13
    }

    codigo_glifo = {v: k for k, v in glifo_codigo.items()}

    resumen_dinamico = {}
    historial_expandido = {}

    for nodo_id, historial in historial_glifos_por_nodo.items():
        nuevo_historial = []
        historial_completo = []
        glifos_validos = []

        for entrada in historial:
            if not isinstance(entrada, (list, tuple)) or len(entrada) != 2:
                print(f"⚠️ Entrada inválida en historial de {nodo_id}: {entrada} — ignorada.")
                continue

            a, b = entrada

            if isinstance(a, str) and a in glifo_codigo:
                glifo, paso = a, b
            elif isinstance(b, str) and b in glifo_codigo:
                glifo, paso = b, a
            elif isinstance(a, int) and a in codigo_glifo:
                paso, codigo = b, a
                glifo = codigo_glifo.get(codigo)
                if glifo is None:
                    print(f"⚠️ Código de glifo inválido en nodo {nodo_id}: {codigo} — omitido.")
                    continue
            elif isinstance(b, int) and b in codigo_glifo:
                paso, codigo = a, b
                glifo = codigo_glifo.get(codigo)
                if glifo is None:
                    print(f"⚠️ Código de glifo inválido en nodo {nodo_id}: {codigo} — omitido.")
                    continue
            else:
                print(f"⚠️ Entrada no reconocida como glifo válido en nodo {nodo_id}: {entrada} — omitida.")
                continue

            try:
                paso_int = int(paso)
                codigo = glifo_codigo[glifo]
                nuevo_historial.append((paso_int, codigo))
                historial_completo.append({
                    "paso": paso_int,
                    "glifo": glifo,
                    "codigo": codigo
                })
                glifos_validos.append(glifo)
            except (ValueError, TypeError):
                print(f"⚠️ Paso no convertible en entero en nodo {nodo_id}: {paso} — omitido.")

        historial_glifos_por_nodo[nodo_id] = nuevo_historial
        historial_expandido[nodo_id] = historial_completo

        if analizar_dinamica and glifos_validos:
            resumen_dinamico[nodo_id] = evaluar_patron_glifico(glifos_validos)

    if analizar_dinamica and expandido:
        return resumen_dinamico, historial_expandido
    elif expandido:
        return historial_expandido
    elif analizar_dinamica:
        return resumen_dinamico
    
def evaluar_patron_glifico(glifos):
    patron = " → ".join(glifos)

    analisis = {
        "ciclos_RE’MESH": glifos.count("RE’MESH"),
        "uso_T’HOL": glifos.count("T’HOL"),
        "uso_Z’HIR": glifos.count("Z’HIR"),
        "latencia_prolongada": any(
            glifos[i] == "SH’A" and glifos[i+1] == "SH’A"
            for i in range(len(glifos) - 1)
        ),
        "inicio_creativo": glifos[0] == "A’L" if glifos else False,
        "coherencia_expansiva": "I’L" in glifos and "VA’L" in glifos,
        "disonancia_sostenida": any(
            glifos[i] == "O’Z" and glifos[i+1] == "O’Z"
            for i in range(len(glifos) - 1)
        ),
        "patron_glifico": patron,
        "tipo_nodal": (
            "creador" if glifos and glifos[0] == "A’L" else
            "mutante" if "Z’HIR" in glifos else
            "colapsante" if glifos.count("RE’MESH") > 2 else
            "expansivo" if "VA’L" in glifos else
            "latente"
        )
    }

    return analisis

def aplicar_glifo(nodo, nodo_id, nombre_glifo, historial_glifos_por_nodo, paso):
    nodo["glifo"] = nombre_glifo
    nodo["estado"] = "silencio" if nombre_glifo == "SH’A" else "activo"

    # Registro en historial global
    if historial_glifos_por_nodo is not None and paso is not None:
        historial_glifos_por_nodo.setdefault(nodo_id, []).append((paso, nombre_glifo))

    # Registro en historial local (para EPIs compuestas)
    if paso is not None:
        if "historial_glifos" not in nodo:
            nodo["historial_glifos"] = []
        nodo["historial_glifos"].append((paso, nombre_glifo))

    # === Transformaciones estructurales por glifo TNFR ===

    if nombre_glifo == "A’L":  # Emisión
        nodo["EPI"] += 0.2
        nodo["Si"] += 0.05
        nodo["νf"] *= 1.05
        nodo["ΔNFR"] *= 0.97

    elif nombre_glifo == "E’N":  # Recepción
        nodo["Si"] += 0.08
        nodo["νf"] *= 0.95
        nodo["θ"] = max(0.0, nodo["θ"] - random.uniform(0.05, 0.15))

    elif nombre_glifo == "I’L":  # Coherencia
        nodo["Si"] += 0.1
        nodo["EPI"] *= 1.05
        nodo["ΔNFR"] *= 0.95

    elif nombre_glifo == "O’Z":  # Disonancia
        nodo["EPI"] *= 0.85
        nodo["ΔNFR"] *= 1.4
        nodo["νf"] *= 1.05
        nodo["Si"] *= 0.9

    elif nombre_glifo == "U’M":  # Acoplamiento
        vecinos = list(G.neighbors(nodo_id))
        if vecinos:
            media_vf = sum(G.nodes[v]["νf"] for v in vecinos) / len(vecinos)
            nodo["νf"] = (nodo["νf"] + media_vf) * 0.5
        nodo["ΔNFR"] *= 0.9

    elif nombre_glifo == "R’A":  # Resonancia
        nodo["Si"] += 0.15
        nodo["EPI"] *= 1.05
        nodo["νf"] *= 1.02

    elif nombre_glifo == "SH’A":  # Silencio
        nodo["estado"] = "silencio"
        nodo["νf"] *= 0.3
        nodo["ΔNFR"] *= 0.1
        nodo["Si"] *= 0.5
        nodo["EPI"] *= 0.9

    elif nombre_glifo == "VA’L":  # Expansión
        nodo["EPI"] *= 1.15
        nodo["Si"] *= 1.08
        nodo["νf"] *= 1.05
        # Nuevo: límite superior preventivo para evitar crecimiento exponencial
        nodo["EPI"] = min(nodo["EPI"], 3.0)

    elif nombre_glifo == "NU’L":  # Contracción
        nodo["EPI"] *= 0.82
        nodo["Si"] *= 0.92
        nodo["νf"] *= 0.92

    elif nombre_glifo == "T’HOL":  # Autoorganización
        nodo["νf"] *= 1.25
        nodo["Si"] *= 1.15
        nodo["θ"] = min(1.0, nodo["θ"] + random.uniform(0.1, 0.2))

    elif nombre_glifo == "Z’HIR":  # Mutación
        nodo["EPI"] += 0.5
        nodo["νf"] *= 1.2
        nodo["θ"] = min(1.0, nodo["θ"] + random.uniform(0.15, 0.3))
        nodo["Si"] *= 1.1

    elif nombre_glifo == "NA’V":  # Nacimiento
        nodo["νf"] *= 1.08
        nodo["ΔNFR"] *= 0.9
        nodo["Si"] += 0.1
        if nodo["estado"] == "silencio":
            nodo["estado"] = "activo"

    elif nombre_glifo == "RE’MESH":  # Recursividad
        nodo["EPI"] = (nodo.get("EPI_prev", nodo["EPI"]) + nodo.get("EPI_prev2", nodo["EPI"])) / 2
        nodo["Si"] *= 0.98
        nodo["νf"] *= 0.98

def emergencia_nodal(nodo, media_vf, std_dNFR):
    return nodo["νf"] > media_vf * 0.9 and abs(nodo["ΔNFR"]) < std_dNFR

def promover_emergente(nodo_id, G, paso, historial_glifos_por_nodo, historia_glifos):
    nodo = G.nodes[nodo_id]

    # Asegurarse de que tiene valores previos
    if "EPI_prev" not in nodo:
        nodo["EPI_prev"] = nodo["EPI"] if np.isfinite(nodo["EPI"]) else 1.0
    if "θ_prev" not in nodo:
        nodo["θ_prev"] = nodo["θ"]

    # Evaluar glifo emergente canónico
    if abs(nodo["EPI"] - nodo["EPI_prev"]) < 0.01 and abs(nodo["ΔNFR"]) < 0.05:
        glifo = "RE’MESH"
    elif abs(nodo.get("θ", 0) - nodo.get("θ_prev", 0)) > 0.2:
        glifo = "Z’HIR"
    elif nodo.get("Si", 0) > 0.8 and nodo.get("glifo") == "U’M":
        glifo = "R’A"
    else:
        glifo = "T’HOL"

    aplicar_glifo(nodo, nodo_id, glifo, historial_glifos_por_nodo, paso)
    historia_glifos.append(f"{paso},{nodo_id},{glifo}")
    nodo["glifo"] = glifo
    nodo["categoria"] = glifo_categoria.get(glifo, "ninguna")
    print(f"Paso {paso}: nodo pulsante '{nodo_id}' promovido a {glifo}")

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
    if abs(nodo["EPI"] - nodo.get("EPI_prev", nodo["EPI"])) > 0.5 and nodo["Si"] > 0.5:
        return "Z’HIR"

    # 7. VA’L – Expansión estructural
    if nodo["Si"] > 0.6 and nodo["EPI"] > 1.2:
        return "VA’L"

    # 8. NU’L – Contracción por exces
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
                Sii, Sij = G.nodes[ni].get("Si", 0), G.nodes[nj].get("Si", 0)
                if abs(θi - θj) < 0.12 and abs(Sii - Sij) < 0.2:
                    grupo_coherente.extend([ni, nj])

        grupo_final = list(set(grupo_coherente))
        if len(grupo_final) >= 3:
            compuestas.append({
                "paso": paso,
                "glifo": glifo,
                "nodos": grupo_final,
                "tipo": clasificar_epi(glifo)
            })

    return compuestas

def clasificar_epi(glifo):
    if glifo in ["I’L", "R’A", "RE’MESH"]:
        return "coherente"
    elif glifo in ["Z’HIR", "VA’L", "NU’L"]:
        return "mutante"
    elif glifo in ["SH’A", "O’Z"]:
        return "disonante"
    else:
        return "otro"

def interpretar_sintaxis_glífica(historial):
    sintaxis = {}
    for nodo, secuencia in historial.items():
        trayecto = [glifo for _, glifo in secuencia]
        transiciones = list(zip(trayecto, trayecto[1:]))
        ciclos_val_nul = sum(
            1 for i in range(len(trayecto)-2)
            if trayecto[i] == "VA’L" and trayecto[i+1] == "NU’L" and trayecto[i+2] == "VA’L"
        )

        tipo = "desconocido"
        if "Z’HIR" in trayecto:
            tipo = "mutante"
        elif "RE’MESH" in trayecto:
            tipo = "recursivo"
        elif ciclos_val_nul >= 2:
            tipo = "pulsante"
        elif trayecto.count("I’L") > 2:
            tipo = "estabilizador"

        sintaxis[nodo] = {
            "trayectoria": trayecto,
            "transiciones": transiciones,
            "mutaciones": trayecto.count("Z’HIR"),
            "colapsos": trayecto.count("SH’A"),
            "ciclos_val_nul": ciclos_val_nul,
            "diversidad_glifica": len(set(trayecto)),
            "tipo_nodal": tipo
        }

    return sintaxis

def aplicar_remesh_red(G, historial_glifos_por_nodo, paso):
    for n in G.nodes:
        nodo = G.nodes[n]
        aplicar_glifo(nodo, n, "RE’MESH", historial_glifos_por_nodo, paso)

def aplicar_remesh_si_estabilizacion_global(G, historial_glifos_por_nodo, historia_glifos, paso):
    if len(G) == 0:
        return

    nodos_estables = 0

    for n in G.nodes:
        nodo = G.nodes[n]
        estabilidad_epi = abs(nodo.get("EPI", 0) - nodo.get("EPI_prev", 0)) < 0.01
        estabilidad_nfr = abs(nodo.get("ΔNFR", 0)) < 0.05
        estabilidad_dEPI = abs(nodo.get("dEPI_dt", 0)) < 0.01
        estabilidad_acel = abs(nodo.get("d2EPI_dt2", 0)) < 0.01

        if all([estabilidad_epi, estabilidad_nfr, estabilidad_dEPI, estabilidad_acel]):
            nodos_estables += 1

    fraccion_estables = nodos_estables / len(G)

    if fraccion_estables > 0.8:
        print(f"\nPaso {paso}: Red global estabilizada ({int(fraccion_estables*100)}%) → activando RE’MESH")
        aplicar_remesh_red(G, historial_glifos_por_nodo, paso)
        for n in G.nodes:
            historial_glifos_por_nodo.setdefault(n, []).append((paso, "RE’MESH"))
            historia_glifos.append(f"{paso},{n},RE’MESH")

print("Iniciando simulación de emergencia...")

# ------------------------- EMERGENCIA -------------------------

def simular_emergencia(G, pasos=1000):
    historia_epi = []
    historia_glifos = ["paso,nodo,glifo"]
    historial_glifos_por_nodo = {}
    G_historia = []
    registro_conexiones = []

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
                    print(f"Paso {paso}: Nodo '{n}' apagado por incoherencia estructural")

            evaluar_si_nodal(nodo, paso)

            if (
                nodo["estado"] == "silencio"
                and abs(nodo["ΔNFR"] - nodo["νf"]) < 0.05
                and nodo.get("Si", 0) > 0.25
                and nodo.get("d2EPI_dt2", 0) > 0.03
                and not reciente_glifo(n, "NA’V", historial_glifos_por_nodo, pasos=6)
            ):
                aplicar_glifo(nodo, n, "NA’V", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},NA’V")
                nodo["estado"] = "activo"
                print(f"Paso {paso}: NA’V activado en nodo '{n}' — transición estructural.")

            if (
                nodo["EPI"] < 0.6
                and abs(nodo["ΔNFR"]) > 0.75
                and nodo["Si"] < 0.25
                and not reciente_glifo(n, "SH’A", historial_glifos_por_nodo, pasos=6)
            ):
                aplicar_glifo(nodo, n, "SH’A", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},SH’A")
                print(f"Paso {paso}: SH’A activado en nodo '{n}' — repliegue vibracional.")
                continue

            if (
                nodo["estado"] == "latente"
                and abs(nodo["ΔNFR"]) < 0.05
                and nodo["Si"] > 0.3
                and not reciente_glifo(n, "E’N", historial_glifos_por_nodo, pasos=10)
            ):
                aplicar_glifo(nodo, n, "E’N", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},E’N")

            if (
                nodo["glifo"] == "I’L"
                and nodo["Si"] > 0.55
                and nodo["νf"] > 1.25
                and abs(nodo["ΔNFR"]) < 0.15  # Baja necesidad de reorganización
                and not reciente_glifo(n, "R’A", historial_glifos_por_nodo, pasos=8)
            ):
                aplicar_glifo(nodo, n, "R’A", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},R’A")
                print(f"Paso {paso}: R’A activado en nodo '{n}' — propagación estructurada.")

            vecinos = list(G.neighbors(n))
            if (
                nodo["estado"] == "activo"
                and vecinos
                and sum(1 for v in vecinos if abs(G.nodes[v]["θ"] - nodo["θ"]) < 0.08) >= 2
                and not reciente_glifo(n, "U’M", historial_glifos_por_nodo, pasos=8)
            ):
                aplicar_glifo(nodo, n, "U’M", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},U’M")

            if (
                abs(nodo.get("d2EPI_dt2", 0)) > 0.25
                and nodo["Si"] > 0.6
                and not reciente_glifo(n, "Z’HIR", historial_glifos_por_nodo, pasos=10)
            ):
                aplicar_glifo(nodo, n, "Z’HIR", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},Z’HIR")
                print(f"Paso {paso}: Z’HIR activado — nodo '{n}' mutó estructuralmente.")

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

            # Activación estructural de VA’L (expansión controlada)
            if (
                nodo["Si"] > 0.8
                and nodo["EPI"] > 1.2
                and abs(nodo["ΔNFR"]) < 0.2
                and nodo.get("dEPI_dt", 0) > 0.15
                and not reciente_glifo(n, "VA’L", historial_glifos_por_nodo, pasos=10)
            ):
                if "expansiones_val" not in nodo:
                    nodo["expansiones_val"] = 0

                if nodo["expansiones_val"] < 3:
                    activar_val_si_estabilidad(n, G, paso, historial_glifos_por_nodo)
                    nodo["expansiones_val"] += 1
                else:
                    aplicar_glifo(nodo, n, "T’HOL", historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{n},T’HOL")
                    print(f"Paso {paso}: nodo '{n}' convertido a T’HOL por sobreexpansión de VA’L")

            if nodo.get("glifo") == "VA’L":
                condiciones_contraccion = (
                    abs(nodo.get("d2EPI_dt2", 0)) < 0.05 and
                    abs(nodo.get("ΔNFR", 0)) < 0.1 and
                    nodo.get("νf", 1.0) < 1.0 and
                    abs(nodo.get("EPI", 0) - nodo.get("EPI_prev", 0)) < 0.01
                )

                if condiciones_contraccion:
                    aplicar_glifo(nodo, n, "NU’L", historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{n},NU’L")
                    nodo["glifo"] = "NU’L"
                    nodo["categoria"] = glifo_categoria.get("NU’L", "ninguna")
                    print(f"Paso {paso}: NU’L activado — nodo '{n}' se contrajo")

            paso_data.append({"nodo": n, "EPI": round(nodo["EPI"], 2)})
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

            # Umbral de bifurcación: si se supera, aplicar T’HOL
            umbral_bifurcacion = 0.12
            if aceleracion > umbral_bifurcacion:
                aplicar_glifo(nodo, n, "T’HOL", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{n},T’HOL")

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
                and not reciente_glifo(n, "VA’L", historial_glifos_por_nodo, pasos=8)
            ):
                activar_val_si_estabilidad(n, G, paso, historial_glifos_por_nodo)

            # Guardar aceleración para graficar más tarde
            if "historial_aceleracion" not in nodo:
                nodo["historial_aceleracion"] = []
            nodo["historial_aceleracion"].append((paso, aceleracion))

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

                if sum(condiciones) >= 3:
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2)
                        registro_conexiones.append({
                            "paso": paso,
                            "accion": "creada",
                            "nodo1": n1,
                            "nodo2": n2,
                            "θ_diff": abs(nodo1["θ"] - nodo2["θ"]),
                            "EPI_diff": abs(nodo1["EPI"] - nodo2["EPI"]),
                            "νf_diff": abs(nodo1["νf"] - nodo2["νf"]),
                            "Si_diff": abs(nodo1["Si"] - nodo2["Si"])
                        })
                        print(f"Paso {paso}: conexión CREADA entre '{n1}' y '{n2}' [...]")
                else:
                    if G.has_edge(n1, n2):
                        G.remove_edge(n1, n2)
                        registro_conexiones.append({
                            "paso": paso,
                            "accion": "eliminada",
                            "nodo1": n1,
                            "nodo2": n2,
                            "θ_diff": abs(nodo1["θ"] - nodo2["θ"]),
                            "EPI_diff": abs(nodo1["EPI"] - nodo2["EPI"]),
                            "νf_diff": abs(nodo1["νf"] - nodo2["νf"]),
                            "Si_diff": abs(nodo1["Si"] - nodo2["Si"])
                        })
                        print(f"Paso {paso}: conexión ELIMINADA entre '{n1}' y '{n2}' [...]")

                if "historial_metabolismo" not in globals():
                    global historial_metabolismo
                    historial_metabolismo = []

                conteo = {
                    "paso": paso,
                    "nodos_totales": len(G.nodes),
                    "activos": sum(1 for n in G.nodes if G.nodes[n]["estado"] == "activo"),
                    "latentes": sum(1 for n in G.nodes if G.nodes[n]["estado"] == "latente"),
                    "silenciosos": sum(1 for n in G.nodes if G.nodes[n]["estado"] == "silencio"),
                    "VAL": sum(1 for n in G.nodes if G.nodes[n].get("glifo") == "VA’L"),
                    "NUL": sum(1 for n in G.nodes if G.nodes[n].get("glifo") == "NU’L"),
                    "aristas": len(G.edges),
                    "Ct": sum(G.nodes[n]["EPI"] for n in G.nodes) / len(G.nodes) if G.nodes else 0
                }

                historial_metabolismo.append(conteo)

                if "historial_pulso" not in globals():
                    global historial_pulso
                    historial_pulso = []

                valores_dEPI = [G.nodes[n].get("dEPI_dt", 0) for n in G.nodes]
                valores_d2EPI = [G.nodes[n].get("d2EPI_dt2", 0) for n in G.nodes]

                prom_dEPI = np.mean(valores_dEPI) if valores_dEPI else 0
                prom_d2EPI = np.mean(valores_d2EPI) if valores_d2EPI else 0

                historial_pulso.append({
                    "paso": paso,
                    "flujo_resonante": prom_dEPI,
                    "bifurcacion_estructural": prom_d2EPI
                })

        # Calcular coherencia total C(t) al final del paso
        C_t = sum(G.nodes[n]["EPI"] for n in G.nodes) / len(G)
        if "historia_Ct" not in globals():
            global historia_Ct
            historia_Ct = []
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
                aplicar_glifo(G.nodes[nodo_id], nodo_id, "RE’MESH", historial_glifos_por_nodo, paso)
                historia_glifos.append(f"{paso},{nodo_id},RE’MESH")

        aplicar_remesh_si_estabilizacion_global(G, historial_glifos_por_nodo, historia_glifos, paso)
        aplicar_remesh_grupal(G, historial_glifos_por_nodo)
        epi_compuestas = detectar_EPIs_compuestas(G)
        if algo_se_mueve(G, historial_glifos_por_nodo, paso):
            historial_macronodos = detectar_macronodos(G, historial_glifos_por_nodo, epi_compuestas, paso)
        else:
            print(f"\n[paso {paso}] No se detectaron cambios relevantes. Macronodos no recalculados.")

        # Evaluar exceso de VA’L y promover reorganización estructural
        for nodo_id, glifos in historial_glifos_por_nodo.items():
            ultimos = [g for _, g in glifos[-6:]]  # últimos 6 glifos del nodo
            if ultimos.count("VA’L") >= 4 and "T’HOL" not in ultimos and "Z’HIR" not in ultimos:
                nodo = G.nodes[nodo_id]
                
                # Se decide el glifo correctivo en función de su Si y ΔNFR
                if nodo["Si"] > 0.5 and abs(nodo["ΔNFR"]) < 0.2:
                    aplicar_glifo(nodo, nodo_id, "T’HOL", historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{nodo_id},T’HOL")
                    print(f"Paso {paso}: Nodo '{nodo_id}' con exceso de VA’L promovido a T’HOL.")
                else:
                    aplicar_glifo(nodo, nodo_id, "Z’HIR", historial_glifos_por_nodo, paso)
                    historia_glifos.append(f"{paso},{nodo_id},Z’HIR")
                    print(f"Paso {paso}: Nodo '{nodo_id}' con exceso de VA’L mutó a Z’HIR por sobreexpansión.")

        porcentaje = int((paso + 1) / total_pasos * 100)
        barra = "█" * (porcentaje // 2) + "-" * (50 - porcentaje // 2)
        nodos_activos = [n for n in G.nodes if G.nodes[n]["estado"] == "activo"]
        print(f"\r [{barra}] {porcentaje}% ({len(nodos_activos)} activos)", end="", flush=True)

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

    with open("10_registro_conexiones.json", "w", encoding="utf-8") as f:
        json.dump(registro_conexiones, f, indent=2, ensure_ascii=False)

    with open("12_historial_macronodos.json", "w", encoding="utf-8") as f:
        json.dump(historial_macronodos, f, indent=2, ensure_ascii=False)

    nodos_pulsantes = detectar_nodos_pulsantes(historial_glifos_por_nodo)

    for nodo_id in nodos_pulsantes:
        nodo = G.nodes[nodo_id]
        historial = historial_glifos_por_nodo.get(nodo_id, [])
        ultimos = [g for _, g in historial][-6:]

        if nodo["glifo"] in ["T’HOL", "Z’HIR", "RE’MESH"]:
            continue  # ya está mutado o recursivo

        nodo = G.nodes[nodo_id]

        # Evaluar emergente canónico
        if abs(nodo["EPI"] - nodo["EPI_prev"]) < 0.01 and abs(nodo["ΔNFR"]) < 0.05:
            glifo = "RE’MESH"
        elif abs(nodo.get("θ", 0) - nodo.get("θ_prev", 0)) > 0.2:
            glifo = "Z’HIR"
        elif nodo.get("Si", 0) > 0.8 and nodo.get("glifo") == "U’M":
            glifo = "R’A"
        else:
            glifo = "T’HOL"

    promover_emergente(nodo_id, G, paso, historial_glifos_por_nodo, historia_glifos)

    return historia_epi, G, epi_compuestas, lecturas, G_historia, historial_glifos_por_nodo

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
    nodo["glifo"] = "NU’L"
    nodo["categoria"] = "contractivo"

    historial_glifos_por_nodo.setdefault(nodo_id, []).append((paso, "NU’L"))

    print(f"Paso {paso}: NU’L aplicado a '{nodo_id}' (contracción por disonancia persistente)")
    
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
        "glifo": "VA’L",
        "categoria": "expansivo",
        "estado": "activo",
        "νf": round(nodo["νf"] * random.uniform(1.0, 1.05), 3),
        "ΔNFR": round(nodo["ΔNFR"] * 0.9, 3),
        "θ": round(nodo["θ"] + random.uniform(-0.01, 0.01), 3),
        "Si": nodo["Si"] * 0.98,
        "historial_glifos": [(paso, "VA’L")],
        "historial_vf": [(paso, nodo["νf"])],
        "historial_dNFR": [(paso, nodo["ΔNFR"])],
        "historial_dEPI_dt": [(paso, nodo.get("dEPI_dt", 0))],
        "historial_Si": [(paso, nodo["Si"])]
    }

    G.add_node(nuevo_id, **nuevo_nodo)
    G.add_edge(nodo_id, nuevo_id)

    historial_glifos_por_nodo.setdefault(nodo_id, []).append((paso, "VA’L"))
    historial_glifos_por_nodo[nuevo_id] = [(paso, "VA’L")]

    nodo["expansiones_val"] = nodo.get("expansiones_val", 0) + 1

    print(f"Paso {paso}: VA’L activado — nodo '{nodo_id}' expandió a '{nuevo_id}'")

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
                g_nodo["glifo"] = "RE’MESH"
                ultimo_paso = historial_glifos_por_nodo[g_id][-1][0] if historial_glifos_por_nodo[g_id] else 0
                historial_glifos_por_nodo[g_id].append((ultimo_paso + 1, "RE’MESH"))
                nodos_aplicados.add(g_id)

    if nodos_aplicados:
        print(f"RE’MESH grupal aplicado en {len(nodos_aplicados)} nodos coherentes.")

def detectar_nodos_pulsantes(historial_glifos_por_nodo, min_ciclos=3):
    nodos_maestros = []
    for nodo_id, eventos in historial_glifos_por_nodo.items():
        glifos = [g for _, g in eventos]
        ciclos = 0
        for i in range(len(glifos) - 1):
            if glifos[i] == "VA’L" and glifos[i+1] == "NU’L":
                ciclos += 1
        if ciclos >= min_ciclos:
            nodos_maestros.append(nodo_id)
    return nodos_maestros

def detectar_macronodos(G, historial_glifos_por_nodo, epi_compuestas, paso, umbral_coherencia=0.05, visualizar=True):   
    historial_macronodos = []
    candidatos = []
    for n in list(G.nodes):
        historial = historial_glifos_por_nodo.get(n, [])
        if len(historial) >= 5:
            glifos_ultimos = [g for _, g in historial[-5:]]
            candidatos.append((n, glifos_ultimos))

    grupos = []
    visitados = set()
    for n1, glifos1 in candidatos:
        if n1 in visitados:
            continue
        grupo = [n1]
        for n2, glifos2 in candidatos:
            if n1 == n2 or n2 in visitados:
                continue
            if glifos1 == glifos2:
                nodo1, nodo2 = G.nodes[n1], G.nodes[n2]
                if abs(nodo1["θ"] - nodo2["θ"]) < 0.1 and abs(nodo1["EPI"] - nodo2["EPI"]) < umbral_coherencia:
                    grupo.append(n2)
        if len(grupo) >= 4:
            grupos.append(grupo)
            visitados.update(grupo)

    log_macros = []
    nuevos_nodos = []
    conexiones = []

    for idx, grupo in enumerate(grupos):
        nombre_macro = f"E_{idx:03d}"
        macro_epi = np.mean([G.nodes[n]["EPI"] for n in grupo])
        macro_vf = np.mean([G.nodes[n]["νf"] for n in grupo])
        macro_Si = np.mean([G.nodes[n]["Si"] for n in grupo])
        macro_theta = np.mean([G.nodes[n]["θ"] for n in grupo])

        nuevo_id = f"{nombre_macro}_N"
        nuevos_nodos.append((nuevo_id, {
            "EPI": macro_epi,
            "νf": macro_vf,
            "Si": macro_Si,
            "θ": macro_theta,
            "ΔNFR": 0.01,
            "glifo": "NA’V",
            "estado": "activo",
            "macro": nombre_macro
        }))

        for nodo_id in grupo:
            historial_glifos_por_nodo[nodo_id].append((paso, 13))  # RE’MESH
            G.nodes[nodo_id]["_marcar_para_remover"] = True

        historial_glifos_por_nodo[nuevo_id] = [
            (paso, "RE’MESH"),
            (paso, "U’M"),
            (paso, "T’HOL")
        ]

        for otro in list(G.nodes):
            if otro == nuevo_id:
                continue
            if G.nodes[otro].get("_marcar_para_remover"):
                continue
            nodo_o = G.nodes[otro]
            condiciones = [
                abs(nodo_o.get("θ", 0) - macro_theta) < 0.1,
                abs(nodo_o.get("EPI", 0) - macro_epi) < 0.2,
                abs(nodo_o.get("νf", 0) - macro_vf) < 0.15,
                abs(nodo_o.get("Si", 0) - macro_Si) < 0.2
            ]
            if sum(condiciones) >= 3:
                conexiones.append((nuevo_id, otro))

        log_macros.append({
            "entidad": nombre_macro,
            "paso": G.graph.get("paso_actual", "NA"),
            "nodo": nuevo_id,
            "EPI": round(macro_epi, 3),
            "νf": round(macro_vf, 3),
            "Si": round(macro_Si, 3),
            "θ": round(macro_theta, 3),
            "subnodos": grupo
        })

    for entrada in epi_compuestas:
        paso = entrada["paso"]
        glifo = entrada["glifo"]
        nodos = entrada["nodos"]

        for nodo in nodos:
            historial_macronodos.append({
                "paso": paso,
                "glifo": glifo,
                "miembros": nodos
            })

        archivo_csv = f"macronodos/macronodo_{paso:04d}_{glifo}.csv"
        with open(archivo_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["nodo", "estado", "θ", "EPI", "Si", "νf", "ΔNFR", "glifo"])
            for nodo in nodos:
                attr = G.nodes[nodo]
                writer.writerow([
                    nodo,
                    attr.get("estado", ""),
                    round(attr.get("θ", 0), 3),
                    round(attr.get("EPI", 0), 3),
                    round(attr.get("Si", 0), 3),
                    round(attr.get("νf", 0), 3),
                    round(attr.get("ΔNFR", 0), 3),
                    attr.get("glifo", "")
                ])

    for n_id in list(G.nodes):
        if G.nodes[n_id].get("_marcar_para_remover"):
            G.remove_node(n_id)

    for nuevo_id, attr in nuevos_nodos:
        G.add_node(nuevo_id, **attr)

    for a, b in conexiones:
        G.add_edge(a, b)

    if len(grupos) > 0:
        print(f"Detectados {len(grupos)} macronodos fractales en red (RE’MESH → U’M → T’HOL). Registro exportado.")

    # Asegurar que todos los nodos tienen los atributos necesarios
    atributos_defecto = {
        "estado": "latente",
        "EPI": 1.0,
        "νf": 1.0,
        "Si": 0.5,
        "θ": 0.0,
        "ΔNFR": 0.0,
        "glifo": "NA’V",
        "categoria": "ninguna"
    }

    for n in G.nodes:
        for k, v in atributos_defecto.items():
            if k not in G.nodes[n]:
                G.nodes[n][k] = v

    return historial_macronodos

def algo_se_mueve(G, historial_glifos_por_nodo, paso, umbral=0.01):
    for nodo in G.nodes:
        datos = G.nodes[nodo]
        
        if datos.get("estado") == "activo":
            return True  # hay actividad
        
        # Comparar cambio reciente de EPI
        epi_actual = datos.get("EPI", 0)
        epi_anterior = datos.get("EPI_prev", epi_actual)
        if abs(epi_actual - epi_anterior) > umbral:
            return True
        
        # Si hay glifos recientes cambiando
        historial = historial_glifos_por_nodo.get(nodo, [])
        if len(historial) >= 5:
            glifos_ultimos = [g for _, g in historial[-5:]]
            if len(set(glifos_ultimos)) > 1:
                return True

    return False

# ------------------------- BLOQUE DE ANÁLISIS FINAL -------------------------

def generar_frames_historia(historia_G, archivo_base="frame", mostrar_etiquetas=False):
    os.makedirs("frames_temp", exist_ok=True)
    anteriores_edges = set()

    print("🧵 Generando frames de la evolución de la red...")
    for i, G in enumerate(tqdm(historia_G, desc="Progreso", unit="frame")):
        plt.clf()
        pos = nx.spring_layout(G, seed=42)

        actuales_edges = set(G.edges())
        nuevas = actuales_edges - anteriores_edges
        rotas = anteriores_edges - actuales_edges
        anteriores_edges = actuales_edges

        glifo_color = {
            "ninguno": "gray", "A’L": "gold", "E’N": "lightblue", "I’L": "blue",
            "O’Z": "red", "U’M": "cyan", "R’A": "orange", "SH’A": "black",
            "VA’L": "magenta", "NU’L": "brown", "T’HOL": "teal",
            "Z’HIR": "darkred", "NA’V": "green", "RE’MESH": "purple"
        }

        colores = [glifo_color.get(G.nodes[n].get("glifo", "ninguno"), "gray") for n in G.nodes]
        tamanos = []
        for n in G.nodes:
            epi = G.nodes[n].get("EPI", 1)
            try:
                epi = float(epi)
                if np.isfinite(epi) and epi >= 0:
                    tamanos.append(min(epi * 700, 4000))
                else:
                    tamanos.append(200)  # tamaño mínimo por defecto
            except Exception:
                tamanos.append(200)

        nx.draw_networkx_nodes(G, pos, node_color=colores, node_size=tamanos, edgecolors="black", linewidths=0.7)
        nx.draw_networkx_edges(G, pos, edgelist=list(actuales_edges - nuevas), alpha=0.1, width=1)

        if nuevas:
            nx.draw_networkx_edges(G, pos, edgelist=list(nuevas), edge_color="lime", width=2.5, style="solid")

        if rotas:
            rotas_validas = [(u, v) for u, v in rotas if u in G.nodes and v in G.nodes]
            nx.draw_networkx_edges(G, pos, edgelist=rotas_validas, edge_color="crimson", width=1.8, style="dashed", alpha=0.6)

        if mostrar_etiquetas:
            etiquetas = {
                n: (n if G.nodes[n].get("glifo", "ninguno") != "ninguno" else "")
                for n in G.nodes
            }
            nx.draw_networkx_labels(G, pos, labels=etiquetas, font_size=7)

        plt.title(f"Paso {i:03d} - evolución de la red")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"frames_temp/{archivo_base}_{i:03d}.png", dpi=200)

def crear_gif(directorio="frames_temp", archivo_salida="11_historia_red.gif", fps=12):

    print("Generando GIF animado de la red...")
    frames = []
    for nombre_archivo in sorted(os.listdir(directorio)):
        if nombre_archivo.endswith(".png"):
            ruta = os.path.join(directorio, nombre_archivo)
            frames.append(imageio.imread(ruta))

    imageio.mimsave(archivo_salida, frames, fps=fps)
    print(f"GIF guardado como '{archivo_salida}'")

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

def evaluar_si_nodal(nodo, paso=None):
    # Factor de estructura vibratoria
    vf = nodo.get("νf", 1.0)
    dNFR = nodo.get("ΔNFR", 0.0)
    theta = nodo.get("θ", 0.5)

    # Glifo actual
    glifo = nodo.get("glifo", "ninguno")

    # Peso estructural simbólico del glifo
    pesos_glifo = {
        "A’L": 1.0,
        "E’N": 1.1,
        "I’L": 1.3,
        "O’Z": 0.6,
        "U’M": 1.2,
        "R’A": 1.5,
        "SH’A": 0.4,
        "VA’L": 1.4,
        "NU’L": 0.8,
        "T’HOL": 1.6,
        "Z’HIR": 1.7,
        "NA’V": 1.0,
        "RE’MESH": 1.3,
        "ninguno": 1.0
    }
    k_glifo = pesos_glifo.get(glifo, 1.0)

    # Cálculo de Si resonante
    Si_nuevo = round((vf / (1 + abs(dNFR))) * k_glifo * theta, 3)

    # Asignar al nodo
    nodo["Si"] = Si_nuevo

    if paso is not None:
        if "historial_Si" not in nodo:
            nodo["historial_Si"] = []
        nodo["historial_Si"].append((paso, Si_nuevo))

    return Si_nuevo

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
        nodos_seleccionados = [
            n for n, d in G.nodes(data=True)
            if any(
                g in [gl for _, gl in d.get("historial_glifos", [])]
                for g in ["NA’V", "I’L", "Z’HIR", "T’HOL", "R’A"]
            )
        ][:100]

    hay_datos = False  # <- Para saber si graficamos al menos una curva

    UMBRAL_MAX = 1000  # puedes ajustarlo según escala esperada

    for nodo_id in nodos_seleccionados:
        historial = G.nodes[nodo_id].get("historial_dEPI_dt", [])
        pasos = [x[0] for x in historial]
        valores = [x[1] for x in historial]

        # Filtrar valores inválidos
        pasos_limpios = []
        valores_limpios = []
        for p, v in zip(pasos, valores):
            if isinstance(v, (int, float)) and np.isfinite(v) and abs(v) < UMBRAL_MAX:
                pasos_limpios.append(p)
                valores_limpios.append(v)

        if len(pasos_limpios) > 1:
            plt.plot(pasos_limpios, valores_limpios, label=f"{nodo_id}")
            hay_datos = True

    if not hay_datos:
        print("⚠️ No hay datos válidos para graficar la dinámica de ∂EPI/∂t.")
        return

    plt.title("Derivada estructural ∂EPI/∂t = νf · ΔNFR")
    plt.xlabel("Paso")
    plt.ylabel("∂EPI/∂t")
    plt.legend(fontsize=6)
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

    def filtrar_historial(historial, umbral_max=1e6):
        pasos, valores = [], []
        for x in historial:
            if (
                isinstance(x[0], (int, float)) and
                isinstance(x[1], (int, float)) and
                np.isfinite(x[0]) and
                np.isfinite(x[1]) and
                abs(x[1]) < umbral_max
            ):
                pasos.append(x[0])
                valores.append(x[1])
        return pasos, valores

    def graficar(historiales, titulo, ylabel, filename):
        plt.figure(figsize=(12, 5))
        datos_validos = False
        for nodo_id in nodos_seleccionados:
            historial = G.nodes[nodo_id].get(historiales, [])
            pasos, valores = filtrar_historial(historial)
            if pasos and valores:
                plt.plot(pasos, valores, label=f"{nodo_id}")
                datos_validos = True
        if datos_validos:
            plt.title(titulo)
            plt.xlabel("Paso")
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True)
            try:
                plt.tight_layout()
            except Exception as e:
                print(f"⚠️ tight_layout falló en {ylabel}: {e}")
            try:
                plt.savefig(filename, dpi=300)
            except Exception as e:
                print(f"⚠️ savefig falló: {e}")
            plt.show()
        else:
            print(f"⚠️ No hay datos válidos para {ylabel}")

    # Ejecutamos las dos gráficas
    graficar("historial_vf", "Evolución de νf (frecuencia estructural)", "νf", "6_frecuencia_estructural.png")
    graficar("historial_dNFR", "Evolución de ΔNFR (gradiente nodal)", "ΔNFR", "7_gradiente_nodal.png")

def graficar_aceleracion_nodal(G, nodos_seleccionados=None):
    if nodos_seleccionados is None:
        # Elegir nodos que hayan activado T’HOL, Z’HIR o RE’MESH
        nodos_seleccionados = [
            n for n, d in G.nodes(data=True)
            if any(
                g in [gl for _, gl in d.get("historial_glifos", [])]
                for g in ["T’HOL", "Z’HIR", "RE’MESH"]
            )
        ][:100]  # limitar a 100 nodos para evitar saturación visual

    for nodo_id in nodos_seleccionados:
        historial = G.nodes[nodo_id].get("historial_aceleracion", [])
        pasos = [x[0] for x in historial]
        valores = [x[1] for x in historial]
        if pasos and valores:
            plt.plot(pasos, valores, label=f"{nodo_id}")

    plt.title("Aceleración estructural ∂²EPI/∂t² por nodo")
    plt.xlabel("Paso")
    plt.ylabel("∂²EPI/∂t²")
    plt.legend(fontsize="small", loc="upper right", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("13_aceleracionnodal.png", dpi=300)
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

def graficar_metabolismo_estructural(historial_metabolismo):
    df = pd.DataFrame(historial_metabolismo)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(df["paso"], df["nodos_totales"], label="Nodos Totales")
    axs[0].plot(df["paso"], df["aristas"], label="Aristas", alpha=0.7)
    axs[0].set_title("Evolución estructural de la red")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(df["paso"], df["activos"], label="Activos")
    axs[1].plot(df["paso"], df["latentes"], label="Latentes")
    axs[1].plot(df["paso"], df["silenciosos"], label="Silenciosos")
    axs[1].set_title("Estados nodales")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(df["paso"], df["VAL"], label="VA’L (Expansión)")
    axs[2].plot(df["paso"], df["NUL"], label="NU’L (Contracción)")
    axs[2].plot(df["paso"], df["Ct"], label="C(t) Coherencia", linestyle="--")
    axs[2].set_title("Metabolismo simbólico")
    axs[2].legend()
    axs[2].grid(True)

    plt.xlabel("Paso de simulación")
    plt.tight_layout()
    plt.savefig("14_metabolismo_estructural.png", dpi=300)
    plt.show()

def graficar_pulso_resonante(historial_pulso):
    df = pd.DataFrame(historial_pulso)

    plt.figure(figsize=(12, 6))
    plt.plot(df["paso"], df["flujo_resonante"], label="∂EPI/∂t — Flujo resonante")
    plt.plot(df["paso"], df["bifurcacion_estructural"], label="∂²EPI/∂t² — Aceleración estructural", linestyle="--")

    plt.title("Pulso fractal resonante de la red")
    plt.xlabel("Paso de simulación")
    plt.ylabel("Magnitud estructural")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("7_pulso_resonante.png", dpi=300)
    plt.show()

def reciente_glifo(nodo_id, glifo_objetivo, historial, pasos=5):
    eventos = historial.get(nodo_id, [])
    if not eventos:
        return False
    try:
        ultimo_paso = int(eventos[-1][0])
    except (ValueError, TypeError):
        return False
    return any(
        g == glifo_objetivo and int(p) >= ultimo_paso - pasos
        for p, g in eventos[-(pasos+1):]
    )

def visualizar_red(G, nodos_emitidos=None):
    if len(G.nodes) == 0:
        print("⚠️ Grafo vacío, no se puede visualizar.")
        return

    pos = nx.spring_layout(G, seed=42)
    nodos_validos = []
    colores = []
    tamaños = []

    for n in G.nodes:
        nodo = G.nodes[n]
        try:
            epi_raw = nodo.get("EPI", 1.0)
            θ_raw = nodo.get("θ", 0.0)

            epi = float(epi_raw) if np.isfinite(epi_raw) and float(epi_raw) >= 0 else 1.0
            θ = float(θ_raw) if np.isfinite(θ_raw) else 0.0

            tamaño = np.sqrt(epi) * 120
            if not np.isfinite(tamaño) or tamaño <= 0:
                continue

            nodos_validos.append(n)
            tamaños.append(tamaño)
            colores.append(θ)
        except Exception as e:
            print(f"⚠️ Nodo inválido {n}: {e}")
            continue

    if not nodos_validos:
        print("⚠️ Ningún nodo válido para dibujar.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    nodos = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodos_validos,
        node_color=colores,
        node_size=tamaños,
        cmap=plt.cm.plasma,
        alpha=0.9,
        ax=ax
    )

    if nodos_emitidos:
        etiquetas = {n: n for n in nodos_emitidos if n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=etiquetas, font_size=8, ax=ax)

    plt.axis("off")
    plt.title("Red Fractal Resonante", fontsize=14)
    plt.savefig("1_red.png", dpi=300)
    plt.close()

archivo_entrada = sys.argv[1] if len(sys.argv) > 1 else "entrada.txt"
G = inicializar_red_desde_archivo(archivo_entrada)
historia_epi, G_final, epi_compuestas, lecturas, G_historia, historial_glifos_por_nodo = simular_emergencia(G)
resumen, historial_dual = normalizar_historial_glifos(historial_glifos_por_nodo, analizar_dinamica=True, expandido=True)
print("Red simbólica analizada y exportada correctamente.")
graficar_dinamica_epi(historia_epi)
historia_si = extraer_dinamica_si(G_historia)
graficar_dinamica_si(historia_si)
graficar_dinamica_dEPI_dt(G)
graficar_frecuencia_y_gradiente(G)
graficar_aceleracion_nodal(G)
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
nodos_emitidos_dict = {n: True for n in emitidos_final}
generar_frames_historia(G_historia)
crear_gif()

print("Red simbólica analizada y exportada correctamente.")
print("Diagnóstico final generado. Proceso completo.")
