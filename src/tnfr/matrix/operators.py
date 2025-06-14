import random

def aplicar_glifo(G, nodo, nodo_id, nombre_glifo, historial_glifos_por_nodo, paso):
    nodo["glifo"] = nombre_glifo
    nodo["estado"] = "silencio" if nombre_glifo == "SHA" else "activo"

    # Preservar valor anterior de θ para detección de mutaciones
    if "θ_prev" not in nodo:
        nodo["θ_prev"] = nodo.get("θ", 0)
    else:
        nodo["θ_prev"] = nodo.get("θ", nodo["θ_prev"])

    # Registro en historial global
    if historial_glifos_por_nodo is not None and paso is not None:
        historial_glifos_por_nodo.setdefault(nodo_id, []).append((paso, nombre_glifo))

    # Registro en historial local (para EPIs compuestas)
    if paso is not None:
        if "historial_glifos" not in nodo:
            nodo["historial_glifos"] = []
        nodo["historial_glifos"].append((paso, nombre_glifo))

    # === Transformaciones estructurales por glifo TNFR ===

    if nombre_glifo == "AL":  # Emisión
        nodo["EPI"] += 0.2
        nodo["Si"] += 0.05
        nodo["νf"] *= 1.05
        nodo["ΔNFR"] *= 0.97

    elif nombre_glifo == "EN":  # Recepción
        nodo["Si"] += 0.08
        nodo["νf"] *= 0.95
        nodo["θ"] = max(0.0, nodo["θ"] - random.uniform(0.05, 0.15))

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
        vecinos = list(G.neighbors(nodo_id))
        if vecinos:
            media_vf = sum(G.nodes[v]["νf"] for v in vecinos) / len(vecinos)
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
        nodo["EPI"] = min(nodo["EPI"], 3.0)  # Límite fijo mientras umbrales no esté disponible

    elif nombre_glifo == "NUL":  # Contracción
        nodo["EPI"] *= 0.82
        nodo["Si"] *= 0.92
        nodo["νf"] *= 0.92

    elif nombre_glifo == "THOL":  # Autoorganización
        nodo["νf"] *= 1.25
        nodo["Si"] *= 1.15
        nodo["θ"] = min(1.0, nodo["θ"] + random.uniform(0.1, 0.2))

    elif nombre_glifo == "ZHIR":  # Mutación
        nodo["EPI"] += 0.5
        nodo["νf"] *= 1.2
        nodo["θ"] = min(1.0, nodo["θ"] + random.uniform(0.15, 0.3))
        nodo["Si"] *= 1.1

    elif nombre_glifo == "NAV":  # Nacimiento
        nodo["νf"] *= 1.08
        nodo["ΔNFR"] *= 0.9
        nodo["Si"] += 0.1
        if nodo["estado"] == "silencio":
            nodo["estado"] = "activo"

    elif nombre_glifo == "REMESH":  # Recursividad
        nodo["EPI"] = (nodo.get("EPI_prev", nodo["EPI"]) + nodo.get("EPI_prev2", nodo["EPI"])) / 2
        nodo["Si"] *= 0.98
        nodo["νf"] *= 0.98

def evaluar_patron_glifico(glifos):
    patron = " → ".join(glifos)

    analisis = {
        "ciclos_REMESH": glifos.count("REMESH"),
        "uso_THOL": glifos.count("THOL"),
        "uso_ZHIR": glifos.count("ZHIR"),
        "latencia_prolongada": any(
            glifos[i] == "SHA" and glifos[i+1] == "SHA"
            for i in range(len(glifos) - 1)
        ),
        "inicio_creativo": glifos[0] == "AL" if glifos else False,
        "coherencia_expansiva": "IL" in glifos and "VAL" in glifos,
        "disonancia_sostenida": any(
            glifos[i] == "OZ" and glifos[i+1] == "OZ"
            for i in range(len(glifos) - 1)
        ),
        "patron_glifico": patron,
        "tipo_nodal": (
            "creador" if glifos and glifos[0] == "AL" else
            "mutante" if "ZHIR" in glifos else
            "colapsante" if glifos.count("REMESH") > 2 else
            "expansivo" if "VAL" in glifos else
            "latente"
        )
    }

    return analisis

def glifo_por_estructura(nodo, G):
    n_id = nodo.get("nodo", None)
    vecinos = list(G.neighbors(n_id)) if n_id else []

    # 1. SHA – Silencio ante alta disonancia
    if nodo["EPI"] < 0.5 and abs(nodo["ΔNFR"]) > 0.8:
        return "SHA"

    # 2. NAV – Activación desde silencio
    if nodo["estado"] == "silencio" and abs(nodo["ΔNFR"] - nodo["νf"]) < 0.05:
        return "NAV"

    # 3. AL – Emisión si es latente y sensible
    if nodo["estado"] == "latente" and nodo["Si"] < 0.2 and nodo["νf"] > 1.0:
        return "AL"

    # 4. EN – Recepción ante apertura sensible
    if nodo["ΔNFR"] > 0.6 and nodo["EPI"] > 1.0 and nodo["Si"] < 0.3:
        return "EN"

    # 5. OZ – Disonancia fuerte
    if abs(nodo["ΔNFR"]) > 1.0 and nodo["EPI"] > 1.0:
        return "OZ"

    # 6. ZHIR – Mutación por cambio abrupto
    if abs(nodo["EPI"] - nodo.get("EPI_prev", nodo["EPI"])) > 0.5 and nodo["Si"] > 0.5:
        return "ZHIR"

    # 7. VAL – Expansión estructural
    if nodo["Si"] > 0.6 and nodo["EPI"] > 1.2:
        return "VAL"

    # 8. NUL – Contracción por exces
    if nodo["EPI"] > 1.3 and nodo["Si"] < 0.4:
        return "NUL"

    # 9. THOL – Autoorganización
    if abs(nodo["EPI"] - nodo["EPI_prev2"]) > 0.2 and abs(nodo["ΔNFR"]) < 0.1:
        return "THOL"

    # 10. IL – Coherencia estable
    if abs(nodo["ΔNFR"]) < 0.05 and abs(nodo["EPI"] - nodo["EPI_prev"]) < 0.05:
        return "IL"

    # 11. RA – Resonancia coherente
    if nodo["glifo"] == "IL" and nodo["Si"] > 0.5 and nodo["νf"] > 1.2:
        return "RA"

    # 12. UM – Acoplamiento con vecinos
    for v in vecinos:
        if abs(nodo["νf"] - G.nodes[v]["νf"]) < 0.05:
            return "UM"

    # 13. REMESH – Recursividad (si ya hay historial)
    hist = nodo.get("historial_glifos", [])
    if (
        len(hist) >= 3
        and hist[-1][1] == hist[-2][1] == hist[-3][1]
        and abs(nodo["EPI"] - nodo["EPI_prev"]) < 0.05
    ):
        return "REMESH"

    return None  # si no se detecta un glifo resonante

def transicion_glifica_canonica(nodo):
    glifo = nodo["glifo"]

    if glifo == "ZHIR":
        if nodo["νf"] > 1.5 and nodo["EPI"] > 2.5:
            return "VAL"
        elif nodo["ΔNFR"] < 0:
            return "RA"
        else:
            return "NAV"

    elif glifo == "IL":
        if nodo["νf"] > 1.2 and nodo["Si"] > 0.4:
            return "RA"

    elif glifo == "OZ":
        if nodo["EPI"] > 2.2 and nodo["Si"] > 0.3:
            return "THOL"

    elif glifo == "NAV":
        if abs(nodo["ΔNFR"]) < 0.1:
            return "IL"

    elif glifo == "RA":
        if nodo["Si"] > 0.6 and nodo["EPI"] > 2.0:
            return "REMESH"

    elif glifo == "VAL":
        if nodo["EPI"] > 3.0 and nodo["Si"] > 0.4:
            return "NUL"

    elif glifo == "AL":
        if nodo["Si"] > 0.3 and nodo["ΔNFR"] < 0.2:
            return "UM"

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

def detectar_EPIs_compuestas(G, umbrales=None):
    # Si no se pasan umbrales, usar valores por defecto
    if umbrales is None:
        umbral_theta = 0.12
        umbral_si = 0.2
    else:
        umbral_theta = umbrales.get('θ_conexion', 0.12)
        umbral_si = umbrales.get('Si_conexion', 0.2)

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
                if abs(θi - θj) < umbral_theta and abs(Sii - Sij) < umbral_si:
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
    if glifo in ["IL", "RA", "REMESH"]:
        return "coherente"
    elif glifo in ["ZHIR", "VAL", "NUL"]:
        return "mutante"
    elif glifo in ["SHA", "OZ"]:
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
            if trayecto[i] == "VAL" and trayecto[i+1] == "NUL" and trayecto[i+2] == "VAL"
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
            "tipo_nodal": tipo
        }

    return sintaxis

def aplicar_remesh_red(G, historial_glifos_por_nodo, paso):
    for n in G.nodes:
        nodo = G.nodes[n]
        aplicar_glifo(G, nodo, n, "REMESH", historial_glifos_por_nodo, paso)

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
        aplicar_remesh_red(G, historial_glifos_por_nodo, paso)
        for n in G.nodes:
            historial_glifos_por_nodo.setdefault(n, []).append((paso, "REMESH"))
            historia_glifos.append(f"{paso},{n},REMESH")

def limpiar_glifo(glifo_raw):
    """
    Limpia glifos que pueden tener comillas adicionales o formato incorrecto
    """
    if not isinstance(glifo_raw, str):
        return str(glifo_raw)
    
    # Remover comillas simples y dobles del inicio y final
    glifo_limpio = glifo_raw.strip().strip("'").strip('"')
    
    # Casos específicos problemáticos
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
        "NA'V": "NAV"
    }
    
    # Buscar coincidencia exacta o parcial
    for glifo_correcto in correcciones.values():
        if glifo_correcto in glifo_limpio or glifo_limpio in glifo_correcto:
            return glifo_correcto
    
    return glifo_limpio

def normalizar_historial_glifos(historial_glifos_por_nodo, analizar_dinamica=False, expandido=False):
    glifo_codigo = {
        "AL": 1, "EN": 2, "IL": 3, "OZ": 4, "UM": 5,
        "RA": 6, "SHA": 7, "VAL": 8, "NUL": 9, "THOL": 10,
        "ZHIR": 11, "NAV": 12, "REMESH": 13
    }
    
    codigo_glifo = {v: k for k, v in glifo_codigo.items()}
    resumen_dinamico = {}
    historial_expandido = {}
    
    for nodo_id, historial in historial_glifos_por_nodo.items():
        nuevo_historial = []
        historial_completo = []
        glifos_validos = []
        
        for entrada in historial:
            # Validación de entrada básica
            if not isinstance(entrada, (list, tuple)) or len(entrada) != 2:
                continue
            
            elemento_a, elemento_b = entrada
            
            # CORRECCIÓN: Lógica simplificada y robusta
            glifo = None
            paso = None
            
            # Caso 1: (paso_int, "glifo_string")
            if isinstance(elemento_a, (int, float)) and isinstance(elemento_b, str):
                glifo_limpio = limpiar_glifo(elemento_b)
                if glifo_limpio in glifo_codigo:
                    paso = elemento_a
                    glifo = glifo_limpio
                            
            # Caso 2: ("glifo_string", paso_int) 
            elif isinstance(elemento_a, str) and isinstance(elemento_b, (int, float)):
                glifo_limpio = limpiar_glifo(elemento_a)
                if glifo_limpio in glifo_codigo:
                    glifo = glifo_limpio
                    paso = elemento_b
            
            # Caso 3: (paso_int, codigo_int)
            elif isinstance(elemento_a, (int, float)) and isinstance(elemento_b, (int, float)):
                if elemento_b in codigo_glifo:
                    paso = elemento_a
                    glifo = codigo_glifo[elemento_b]
                elif elemento_a in codigo_glifo:
                    paso = elemento_b
                    glifo = codigo_glifo[elemento_a]
            
            # Validación final
            if glifo is None or paso is None:
                continue
            
            # Conversión segura de paso a entero
            try:
                paso_int = int(float(paso))  # Doble conversión para manejar floats
                if paso_int < 0:
                    continue
            except (ValueError, TypeError) as e:
                continue
            
            # Validación del glifo
            glifo_final = limpiar_glifo(glifo)
            if glifo_final not in glifo_codigo:
                continue
            glifo = glifo_final
            
            # Agregar entrada válida
            codigo = glifo_codigo[glifo]
            nuevo_historial.append((paso_int, codigo))
            historial_completo.append({
                "paso": paso_int,
                "glifo": glifo,
                "codigo": codigo
            })
            glifos_validos.append(glifo)
        
        # Actualizar historial procesado
        historial_glifos_por_nodo[nodo_id] = nuevo_historial
        historial_expandido[nodo_id] = historial_completo
        
        # Análisis dinámico si se solicita
        if analizar_dinamica and glifos_validos:
            resumen_dinamico[nodo_id] = evaluar_patron_glifico(glifos_validos)
    
    # Retornar según parámetros
    if analizar_dinamica and expandido:
        return resumen_dinamico, historial_expandido
    elif expandido:
        return historial_expandido
    elif analizar_dinamica:
        return resumen_dinamico
    
    return historial
    
__all__ = [
    'aplicar_glifo',
    'evaluar_patron_glifico',
    'glifo_por_estructura',
    'transicion_glifica_canonica',
    'acoplar_nodos',
    'detectar_EPIs_compuestas',
    'clasificar_epi',
    'normalizar_historial_glifos',
    'interpretar_sintaxis_glífica',
    'aplicar_remesh_red',
    'aplicar_remesh_si_estabilizacion_global',
    'limpiar_glifo',
]

    
