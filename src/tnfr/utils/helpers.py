from tnfr.matrix.operators import aplicar_glifo
from tnfr.constants import glifo_categoria
import numpy as np

def emergencia_nodal(nodo, media_vf, std_dNFR):
    return nodo["νf"] > media_vf * 0.9 and abs(nodo["ΔNFR"]) < std_dNFR

def promover_emergente(nodo_id, G, paso, historial_glifos_por_nodo, historia_glifos):
    if nodo_id not in G:
        return
    nodo = G.nodes[nodo_id]

    # Asegurarse de que tiene valores previos
    if "EPI_prev" not in nodo:
        nodo["EPI_prev"] = nodo["EPI"] if np.isfinite(nodo["EPI"]) else 1.0
    if "θ_prev" not in nodo:
        nodo["θ_prev"] = nodo["θ"]

    # Evaluar glifo emergente canónico
    if abs(nodo["EPI"] - nodo["EPI_prev"]) < 0.01 and abs(nodo["ΔNFR"]) < 0.05:
        glifo = "REMESH"
    elif abs(nodo.get("θ", 0) - nodo.get("θ_prev", 0)) > 0.2:
        glifo = "ZHIR"
    elif nodo.get("Si", 0) > 0.8 and nodo.get("glifo") == "UM":
        glifo = "RA"
    else:
        glifo = "THOL"

    aplicar_glifo(G, nodo, nodo_id, glifo, historial_glifos_por_nodo, paso)
    historia_glifos.append(f"{paso},{nodo_id},{glifo}")
    nodo["glifo"] = glifo
    nodo["categoria"] = glifo_categoria.get(glifo, "ninguna")

def detectar_nodos_pulsantes(historial_glifos_por_nodo, min_ciclos=3):
    nodos_maestros = []
    for nodo_id, eventos in historial_glifos_por_nodo.items():
        glifos = [g for _, g in eventos]
        ciclos = 0
        for i in range(len(glifos) - 1):
            if glifos[i] == "VAL" and glifos[i+1] == "NUL":
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
        # Determinar glifo predominante
        glifos_grupo = []
        for nodo in grupo:
            glifos_grupo += [g for _, g in historial_glifos_por_nodo.get(nodo, [])]
        if glifos_grupo:
            glifo_predominante = max(set(glifos_grupo), key=glifos_grupo.count)
        else:
            glifo_predominante = "X"

        # Determinar EPI media categorizada
        macro_epi = np.mean([G.nodes[n]["EPI"] for n in grupo])
        if macro_epi > 2.0:
            epi_cat = "H"
        elif macro_epi > 1.2:
            epi_cat = "M"
        else:
            epi_cat = "L"

        nombre_macro = f"M_{glifo_predominante}_{epi_cat}_{idx:02d}"

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
            "glifo": "NAV",
            "estado": "activo",
            "macro": nombre_macro
        }))

        for nodo_id in grupo:
            historial_glifos_por_nodo[nodo_id].append((paso, 13))  # REMESH
            G.nodes[nodo_id]["_marcar_para_remover"] = True

        historial_glifos_por_nodo[nuevo_id] = [
            (paso, "REMESH"),
            (paso, "UM"),
            (paso, "THOL")
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

    for n_id in list(G.nodes):
        if G.nodes[n_id].get("_marcar_para_remover"):
            G.remove_node(n_id)

    for nuevo_id, attr in nuevos_nodos:
        G.add_node(nuevo_id, **attr)

    for a, b in conexiones:
        G.add_edge(a, b)

    # Asegurar que todos los nodos tienen los atributos necesarios
    atributos_defecto = {
        "estado": "latente",
        "EPI": 1.0,
        "νf": 1.0,
        "Si": 0.5,
        "θ": 0.0,
        "ΔNFR": 0.0,
        "glifo": "NAV",
        "categoria": "ninguna"
    }

    for n in G.nodes:
        for k, v in atributos_defecto.items():
            if k not in G.nodes[n]:
                G.nodes[n][k] = v

    macronodes_info = {
        'nodos': [nuevo_id for nuevo_id, _ in nuevos_nodos],
        'conexiones': conexiones
    }

    return historial_macronodos, macronodes_info

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

def extraer_dinamica_si(G_historia):
    historia_si = []
    for paso, G in enumerate(G_historia):
        paso_data = []
        for n in G.nodes:
            paso_data.append({
                "nodo": n, 
                "paso": paso, 
                "Si": round(G.nodes[n]["Si"], 3)
            })
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

def obtener_nodos_emitidos(G):
    if len(G.nodes) == 0:
        return [], []
    
    # Extraer nodos emitidos por coherencia estructural
    emitidos_final = [
        n for n in G.nodes
        if G.nodes[n]["glifo"] != "ninguno"
        and G.nodes[n].get("categoria", "ninguna") not in ["sin categoría", "ninguna"]
    ]
    
    # Generar resultado detallado con información completa
    resultado_detallado = []
    for n in emitidos_final:
        nodo = G.nodes[n]
        entrada = {
            "nodo": n,
            "glifo": nodo["glifo"],
            "EPI": round(nodo["EPI"], 4),
            "Si": round(nodo.get("Si", 0), 4),
            "ΔNFR": round(nodo.get("ΔNFR", 0), 4),
            "θ": round(nodo.get("θ", 0), 4),
            "νf": round(nodo.get("νf", 1.0), 4),
            "categoria": nodo.get("categoria", "ninguna")
        }
        resultado_detallado.append(entrada)
    
    return emitidos_final, resultado_detallado

def exportar_nodos_emitidos(G, emitidos_final=None, archivo='nodos_emitidos.json'):
    try:
        # Obtener nodos emitidos si no se proporcionan
        if emitidos_final is None:
            emitidos_final, _ = obtener_nodos_emitidos(G)
        
        if not emitidos_final:
            return {
                'exitosa': False,
                'razon': 'No hay nodos emitidos para exportar',
                'nodos_exportados': 0
            }
        
        return {
            'exitosa': True,
            'archivo': archivo,
            'nodos_exportados': len(emitidos_final)
        }
        
    except Exception as e:
        return {
            'exitosa': False,
            'razon': f"Error durante exportación: {str(e)}",
            'nodos_exportados': 0
        }
    
def crear_diccionario_nodos_emitidos(emitidos_final):
    return {n: True for n in emitidos_final}

# Al final del archivo
__all__ = [
    'emergencia_nodal',
    'promover_emergente',
    'detectar_nodos_pulsantes',
    'detectar_macronodos',
    'algo_se_mueve',
    'obtener_nodos_emitidos',
    'evaluar_si_nodal',
    'reciente_glifo',
    'algo_se_mueve',
    'extraer_dinamica_si',
    'exportar_nodos_emitidos',
    'crear_diccionario_nodos_emitidos'
]
