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
import math
import networkx as nx
import random
import json
import csv
import pandas as pd
import numpy as np
import sys
import re
import string
import os
from collections import defaultdict, Counter, deque
from math import isnan
from dataclasses import dataclass
from typing import TypedDict, List, Dict, Any, Optional

class TNFRNode(TypedDict):
    id: str
    EPI: float
    νf: float
    Si: float
    θ: float
    ΔNFR: float
    glifo: str
    estado: str
    Wi_t: Any  # np.ndarray
    fase: float
    simetria_interna: float
    metadata: Dict[str, Any]

# GESTIÓN TEMPORAL TOPOLÓGICA TNFR # 

class TemporalCoordinatorTNFR:
    """
    Coordinador temporal canónico que gestiona tiempo topológico variable
    según frecuencias estructurales νf de cada NFR y principios de entrainment.
    """
    
    def __init__(self, sincronizacion_global=True, pulsos_reorganizacion=50):
        # Configuración temporal canónica
        self.sincronizacion_global = sincronizacion_global
        self.frecuencia_pulsos = pulsos_reorganizacion
        self.tiempo_topologico = 0.0
        
        # Estados temporales de nodos
        self.cronometros_nodales = {}  # tiempo local de cada nodo
        self.fases_sincronizacion = {}  # fase temporal de cada nodo
        self.ultimas_activaciones = {}  # última activación de cada nodo
        
        # Historial de sincronización
        self.historial_entrainment = []
        self.historial_coherencia_temporal = []
        
        # Cola de eventos temporales
        self.cola_eventos = []  # [(tiempo_activacion, nodo_id, tipo_evento)]
        
        # Parámetros de resonancia temporal
        self.umbral_resonancia = 0.15  # diferencia máxima en νf para resonancia
        self.factor_aceleracion = 1.8  # aceleración temporal por coherencia alta
        
    def calcular_paso_temporal_nodal(self, nodo, paso_global):
        # Validación preventiva de parámetros
        vf_nodo = nodo.get("νf", 1.0)
        if not np.isfinite(vf_nodo) or vf_nodo <= 0:
            vf_nodo = 1.0
            nodo["νf"] = 1.0  # Corrección in-situ
        
        Si_nodo = nodo.get("Si", 0.5)
        if not np.isfinite(Si_nodo):
            Si_nodo = 0.5
            nodo["Si"] = 0.5
    
        vf_nodo = nodo.get("νf", 1.0)
        Si_nodo = nodo.get("Si", 0.5)
        theta_nodo = nodo.get("θ", 0.5)
        estado = nodo.get("estado", "latente")
        
        # Paso base según frecuencia estructural (inversa de νf)
        # Alta frecuencia = pasos más pequeños (más actividad)
        paso_base = 1.0 / max(0.1, vf_nodo)
        
        # Factor de coherencia: mayor Si permite pasos más largos (estabilidad)
        factor_coherencia = 0.5 + 0.5 * Si_nodo
        
        # Factor de activación: nodos activos necesitan pasos más pequeños
        factor_activacion = {
            "activo": 0.7,      # pasos más pequeños para actividad
            "latente": 1.0,     # pasos normales
            "silencio": 1.5,    # pasos más grandes en silencio
            "inactivo": 2.0     # pasos muy grandes si inactivo
        }.get(estado, 1.0)
        
        # Factor de umbral: cerca de bifurcación = pasos pequeños
        factor_umbral = 1.0 - 0.3 * min(1.0, theta_nodo)
        
        # Combinar todos los factores
        paso_temporal = paso_base * factor_coherencia * factor_activacion * factor_umbral
        
        # Limitar al rango [0.1, 5.0] para evitar extremos
        paso_temporal = max(0.1, min(5.0, paso_temporal))
        
        return paso_temporal
    
    def detectar_nodos_resonantes(self, G):
        """
        Detecta grupos de nodos con frecuencias νf compatibles para entrainment.
        """
        nodos_por_frecuencia = defaultdict(list)
        
        # Agrupar nodos por bandas de frecuencia
        for nodo_id, nodo_data in G.nodes(data=True):
            vf = nodo_data.get("νf", 1.0)
            if np.isfinite(vf) and vf > 0 and np.isfinite(self.umbral_resonancia):
                try:
                    ratio = vf / self.umbral_resonancia
                    if np.isfinite(ratio) and abs(ratio) < 1e6:  # Límite de seguridad
                        banda_freq = round(ratio) * self.umbral_resonancia
                    else:
                        banda_freq = self.umbral_resonancia  # Valor por defecto seguro
                except (ValueError, OverflowError):
                    banda_freq = self.umbral_resonancia
            else:
                banda_freq = self.umbral_resonancia  # Manejo de casos inválidos
            nodos_por_frecuencia[banda_freq].append(nodo_id)
        
        # Identificar grupos resonantes (2+ nodos en misma banda)
        grupos_resonantes = []
        for banda, nodos in nodos_por_frecuencia.items():
            if len(nodos) >= 2:
                # Verificar coherencia estructural dentro del grupo
                coherencias = [G.nodes[n].get("Si", 0) for n in nodos]
                if np.mean(coherencias) > 0.4:  # grupo coherente
                    grupos_resonantes.append({
                        'banda_frecuencia': banda,
                        'nodos': nodos,
                        'coherencia_grupal': np.mean(coherencias),
                        'tamaño': len(nodos)
                    })
        
        return grupos_resonantes
    
    def sincronizar_grupo_resonante(self, G, grupo):
        """
        Sincroniza temporalmente un grupo de nodos resonantes mediante entrainment.
        """
        nodos = grupo['nodos']
        banda_freq = grupo['banda_frecuencia']
        
        # Calcular fase de sincronización grupal
        fases_actuales = [self.fases_sincronizacion.get(n, 0.0) for n in nodos]
        fase_promedio = np.mean(fases_actuales)
        
        # Factor de atracción hacia sincronización
        for nodo_id in nodos:
            nodo = G.nodes[nodo_id]
            fase_actual = self.fases_sincronizacion.get(nodo_id, 0.0)
            
            # Calcular corrección de fase hacia el promedio grupal
            diferencia_fase = fase_promedio - fase_actual
            factor_correccion = 0.1 * nodo.get("Si", 0.5)  # más Si = más atraído
            
            # Aplicar corrección suave
            nueva_fase = fase_actual + factor_correccion * diferencia_fase
            self.fases_sincronizacion[nodo_id] = nueva_fase % (2 * np.pi)
            
            # Ajustar cronómetro nodal para sincronización
            ajuste_temporal = np.sin(diferencia_fase) * 0.05
            cronometro_actual = self.cronometros_nodales.get(nodo_id, 0.0)
            self.cronometros_nodales[nodo_id] = cronometro_actual + ajuste_temporal
        
        return len(nodos)  # cantidad de nodos sincronizados
    
    def generar_pulso_reorganizacion_global(self, G, paso_global):
        """
        Genera pulso de reorganización global que sincroniza toda la red.
        """
        if paso_global % self.frecuencia_pulsos != 0:
            return False
        
        # Calcular coherencia global actual
        EPIs = [G.nodes[n].get("EPI", 1.0) for n in G.nodes]
        coherencia_global = np.mean(EPIs)
        
        # Intensidad del pulso basada en necesidad de reorganización
        variabilidad_EPI = np.std(EPIs)
        intensidad_pulso = min(1.0, variabilidad_EPI / coherencia_global)
        
        # Aplicar pulso a todos los nodos
        nodos_afectados = 0
        for nodo_id, nodo_data in G.nodes(data=True):
            # Reset parcial del cronómetro según intensidad
            cronometro_actual = self.cronometros_nodales.get(nodo_id, 0.0)
            ajuste = intensidad_pulso * 0.2 * random.uniform(-1, 1)
            self.cronometros_nodales[nodo_id] = cronometro_actual + ajuste
            
            # Sincronizar fases hacia coherencia global
            fase_actual = self.fases_sincronizacion.get(nodo_id, 0.0)
            # Fase objetivo basada en frecuencia del nodo
            vf = nodo_data.get("νf", 1.0)
            tiempo_normalizado = self.tiempo_topologico % (4 * np.pi)  # Ciclo de normalización
            if np.isfinite(vf) and np.isfinite(tiempo_normalizado):
                fase_objetivo = (vf * tiempo_normalizado) % (2 * np.pi)
            else:
                fase_objetivo = 0.0  # Valor seguro por defecto
            diferencia = fase_objetivo - fase_actual
            self.fases_sincronizacion[nodo_id] = fase_actual + 0.3 * diferencia
            
            nodos_afectados += 1
        
        return True
    
    def calcular_simultaneidad_eventos(self, G, eventos_candidatos):
        """
        Determina qué eventos pueden ocurrir simultáneamente basado en coherencia.
        """
        if len(eventos_candidatos) <= 1:
            return eventos_candidatos
        
        eventos_simultaneos = []
        eventos_procesados = set()
        
        for i, (tiempo_i, nodo_i, evento_i) in enumerate(eventos_candidatos):
            if i in eventos_procesados:
                continue
                
            grupo_simultaneo = [(tiempo_i, nodo_i, evento_i)]
            eventos_procesados.add(i)
            
            # Buscar eventos compatibles para simultaneidad
            for j, (tiempo_j, nodo_j, evento_j) in enumerate(eventos_candidatos[i+1:], i+1):
                if j in eventos_procesados:
                    continue
                    
                # Verificar criterios de simultaneidad
                diferencia_temporal = abs(tiempo_i - tiempo_j)
                if diferencia_temporal > 0.1:  # demasiado separados en tiempo
                    continue
                
                # Verificar coherencia estructural entre nodos
                nodo_data_i = G.nodes[nodo_i]
                nodo_data_j = G.nodes[nodo_j]
                
                diferencia_vf = abs(nodo_data_i.get("νf", 1) - nodo_data_j.get("νf", 1))
                diferencia_Si = abs(nodo_data_i.get("Si", 0) - nodo_data_j.get("Si", 0))
                
                # Criterios de compatibilidad para simultaneidad
                if (diferencia_vf < self.umbral_resonancia and 
                    diferencia_Si < 0.3 and
                    len(grupo_simultaneo) < 5):  # máximo 5 eventos simultáneos
                    
                    grupo_simultaneo.append((tiempo_j, nodo_j, evento_j))
                    eventos_procesados.add(j)
            
            eventos_simultaneos.append(grupo_simultaneo)
        
        return eventos_simultaneos
    
    def avanzar_tiempo_topologico(self, G, paso_global):
        """
        Función principal que avanza el tiempo topológico de la red.
        """
        eventos_este_paso = []
        grupos_resonantes = self.detectar_nodos_resonantes(G)

        if self.tiempo_topologico > 1e6 or not np.isfinite(self.tiempo_topologico):
            self.tiempo_topologico = self.tiempo_topologico % (8 * np.pi)  # Normalizar
        if not np.isfinite(self.tiempo_topologico):
            self.tiempo_topologico = 0.0  # Reset completo si persiste NaN
        
        # Procesar cada nodo con su tiempo topológico individual
        for nodo_id, nodo_data in G.nodes(data=True):
            # Inicializar cronómetro si es necesario
            if nodo_id not in self.cronometros_nodales:
                self.cronometros_nodales[nodo_id] = 0.0
                self.fases_sincronizacion[nodo_id] = random.uniform(0, 2*np.pi)
            
            # Calcular paso temporal para este nodo
            paso_nodal = self.calcular_paso_temporal_nodal(nodo_data, paso_global)
            
            # Avanzar cronómetro nodal
            self.cronometros_nodales[nodo_id] += paso_nodal
            
            # Actualizar fase de sincronización
            vf = nodo_data.get("νf", 1.0)
            incremento_fase = 2 * np.pi * paso_nodal * vf
            self.fases_sincronizacion[nodo_id] = (self.fases_sincronizacion[nodo_id] + incremento_fase) % (2 * np.pi)
            
            # Verificar si el nodo debe activarse en este paso
            tiempo_desde_activacion = self.cronometros_nodales[nodo_id] - self.ultimas_activaciones.get(nodo_id, 0)
            
            # Umbral de activación basado en frecuencia y fase
            umbral_activacion = 1.0 / max(0.1, vf)  # período de activación
            
            if tiempo_desde_activacion >= umbral_activacion:
                eventos_este_paso.append((self.cronometros_nodales[nodo_id], nodo_id, "activacion_temporal"))
                self.ultimas_activaciones[nodo_id] = self.cronometros_nodales[nodo_id]

        # Control de desbordamiento de cronómetros
        for nodo_id in self.cronometros_nodales:
            if self.cronometros_nodales[nodo_id] > 1e4:
                self.cronometros_nodales[nodo_id] = self.cronometros_nodales[nodo_id] % 100.0

        # Sincronizar grupos resonantes
        nodos_sincronizados = 0
        for grupo in grupos_resonantes:
            nodos_sincronizados += self.sincronizar_grupo_resonante(G, grupo)
        
        # Generar pulso de reorganización global si corresponde
        pulso_global = self.generar_pulso_reorganizacion_global(G, paso_global)
        
        # Calcular eventos simultáneos
        grupos_simultaneos = self.calcular_simultaneidad_eventos(G, eventos_este_paso)
        
        # Avanzar tiempo topológico global
        incremento_global = np.mean([self.calcular_paso_temporal_nodal(G.nodes[n], paso_global) for n in G.nodes])
        self.tiempo_topologico += incremento_global
        
        # Registrar estadísticas temporales
        coherencia_temporal = self.calcular_coherencia_temporal(G)
        self.historial_coherencia_temporal.append((paso_global, coherencia_temporal))
        
        # Registrar información de entrainment
        self.historial_entrainment.append({
            'paso': paso_global,
            'grupos_resonantes': len(grupos_resonantes),
            'nodos_sincronizados': nodos_sincronizados,
            'eventos_simultaneos': len([g for g in grupos_simultaneos if len(g) > 1]),
            'pulso_global': pulso_global,
            'coherencia_temporal': coherencia_temporal
        })
        
        return {
            'tiempo_topologico': self.tiempo_topologico,
            'grupos_resonantes': grupos_resonantes,
            'eventos_simultaneos': grupos_simultaneos,
            'estadisticas': self.historial_entrainment[-1]
        }
    
    def calcular_coherencia_temporal(self, G):
        """
        Calcula la coherencia temporal global de la red.
        """
        if len(G.nodes) == 0:
            return 0.0
        
        # Coherencia basada en sincronización de fases
        fases = [self.fases_sincronizacion.get(n, 0) for n in G.nodes]
        
        # Calcular parámetro de orden de Kuramoto
        suma_compleja = sum(np.exp(1j * fase) for fase in fases)
        parametro_orden = abs(suma_compleja) / len(fases)
        
        # Coherencia basada en distribución de cronómetros
        cronometros = [self.cronometros_nodales.get(n, 0) for n in G.nodes]
        variabilidad_cronometros = np.std(cronometros) / (np.mean(cronometros) + 0.1)
        coherencia_cronometros = 1.0 / (1.0 + variabilidad_cronometros)
        
        # Combinar ambas métricas
        coherencia_temporal = 0.6 * parametro_orden + 0.4 * coherencia_cronometros
        
        return coherencia_temporal

def inicializar_coordinador_temporal_canonico():
    """
    Inicializa el coordinador temporal canónico para OntoSim.
    """
    return TemporalCoordinatorTNFR(
        sincronizacion_global=True,
        pulsos_reorganizacion=75  # pulso cada 75 pasos
    )

def integrar_tiempo_topologico_en_simulacion(G, paso, coordinador_temporal):
    """
    Función de integración que debe llamarse en cada paso de simular_emergencia().
    Reemplaza la gestión temporal lineal por tiempo topológico canónico.
    """
    resultado_temporal = coordinador_temporal.avanzar_tiempo_topologico(G, paso)
    
    # Aplicar efectos temporales a los nodos
    for nodo_id, nodo_data in G.nodes(data=True):
        # Obtener información temporal del nodo
        cronometro = coordinador_temporal.cronometros_nodales.get(nodo_id, 0)
        fase = coordinador_temporal.fases_sincronizacion.get(nodo_id, 0)
        
        # Modular parámetros TNFR según tiempo topológico
        modulacion_temporal = 1.0 + 0.1 * np.sin(fase)  # modulación suave
        
        # Aplicar modulación a νf (retroalimentación temporal)
        vf_actual = nodo_data.get("νf", 1.0)
        nodo_data["νf"] = vf_actual * modulacion_temporal
        
        # Registrar información temporal en el nodo
        nodo_data["cronometro_topologico"] = cronometro
        nodo_data["fase_temporal"] = fase
        nodo_data["ultima_sincronizacion"] = paso
    
    return resultado_temporal

# ------------------------- INICIALIZACIÓN -------------------------

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

def _deben_conectarse_canonico(n1: dict, n2: dict) -> bool:
    """Mejora la lógica existente con umbral áureo"""
    phi = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618
    
    diferencia_vf = abs(n1.get('νf', 1) - n2.get('νf', 1))
    diferencia_fase = abs(n1.get('fase', 0) - n2.get('fase', 0)) % (2 * math.pi)
    
    return (diferencia_vf < 0.01 * phi and 
            diferencia_fase < math.pi / 2)

glifo_categoria = {
    "AL": "emisión",
    "EN": "recepción",
    "IL": "coherencia",
    "OZ": "disonancia",
    "UM": "acoplamiento",
    "RA": "resonancia",
    "SHA": "silencio",
    "VAL": "expansión",
    "NUL": "contracción",
    "THOL": "autoorganización",
    "ZHIR": "mutación",
    "NAV": "nacimiento",
    "REMESH": "recursividad"
}

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

# =========================================================================================
# SISTEMA DE UMBRALES DINÁMICOS
# =========================================================================================

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

def evaluar_condiciones_emergencia_dinamica(nodo, umbrales, campo_coherencia):
    """
    Evalúa si un nodo cumple condiciones de emergencia con umbrales dinámicos.

    Args:
        nodo: Diccionario con parámetros del nodo
        umbrales: Umbrales calculados dinámicamente
        campo_coherencia: C(t) actual de la red

    Returns:
        tuple: (puede_emerger, razon_rechazo)
    """
    # Verificación de coherencia estructural mínima
    if nodo.get("EPI", 0) < umbrales['EPI_min_coherencia']:
        return False, f"EPI insuficiente: {nodo.get('EPI', 0):.3f} < {umbrales['EPI_min_coherencia']:.3f}"

    # Verificación de frecuencia resonante
    if nodo.get("νf", 0) < 0.3:  # mínimo absoluto para vibración
        return False, f"Frecuencia demasiado baja: {nodo.get('νf', 0):.3f}"

    # Verificación de compatibilidad con campo de coherencia
    fase_nodo = nodo.get("fase", 0.5)
    if abs(fase_nodo - 0.0) > 0.7 and campo_coherencia > 1.2:
        return False, f"Disonancia con campo: fase={fase_nodo:.3f}, C(t)={campo_coherencia:.3f}"

    # Verificación de gradiente nodal dentro de límites
    ΔNFR = abs(nodo.get("ΔNFR", 0))
    if ΔNFR > umbrales['bifurcacion_gradiente']:
        return False, f"Gradiente excesivo: {ΔNFR:.3f} > {umbrales['bifurcacion_gradiente']:.3f}"

    return True, "Condiciones de emergencia cumplidas"

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

def evaluar_activacion_glifica_dinamica(nodo, umbrales, vecinos_data=None):
    """
    Evalúa qué glifo debería activarse basado en umbrales dinámicos.

    Args:
        nodo: Datos del nodo
        umbrales: Umbrales dinámicos calculados
        vecinos_data: Lista de datos de nodos vecinos

    Returns:
        str or None: Glifo a activar o None si no hay activación
    """
    # Z'HIR - Mutación por umbral de cambio estructural
    θ_actual = nodo.get("θ", 0)
    θ_prev = nodo.get("θ_prev", θ_actual)

    if abs(θ_actual - θ_prev) > umbrales['θ_mutacion']:
        return "ZHIR"

    # SH'A - Colapso por pérdida de coherencia
    if (nodo.get("EPI", 0) < umbrales['EPI_min_coherencia'] and 
        abs(nodo.get("ΔNFR", 0)) > umbrales['θ_colapso']):
        return "SHA"

    # T'HOL - Autoorganización por aceleración estructural
    aceleracion = abs(nodo.get("d2EPI_dt2", 0))
    if aceleracion > umbrales.get('bifurcacion_aceleracion', 0.15):
        return "THOL"

    # R'A - Resonancia con vecinos (requiere datos de vecinos)
    if vecinos_data and len(vecinos_data) > 0:
        θ_vecinos = [v.get("θ", 0) for v in vecinos_data]
        resonancia_promedio = sum(abs(θ_actual - θ_v) for θ_v in θ_vecinos) / len(θ_vecinos)

        if resonancia_promedio < umbrales['θ_conexion'] * 0.5:  # muy sincronizado
            return "RA"
        
    if (nodo.get("EPI", 0) < umbrales.get('EPI_min_coherencia', 0.4) and
        abs(nodo.get("ΔNFR", 0)) > umbrales.get('θ_colapso', 0.45)):
        return "SHA"

    return None

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

# Sistema de Bifurcaciones Estructurales Múltiples

# Clase para representar una trayectoria de bifurcación
@dataclass
class TrayectoriaBifurcacion:
    """Representa una trayectoria específica en una bifurcación estructural"""
    id: str
    tipo: str
    secuencia_glifica: List[str]
    parametros_iniciales: Dict[str, float]
    viabilidad: float = 1.0
    pasos_completados: int = 0
    activa: bool = True
    convergencia_objetivo: Optional[str] = None

# Clase para gestionar espacios de bifurcación
@dataclass
class EspacioBifurcacion:
    """Representa el espacio completo de una bifurcación con múltiples trayectorias"""
    nodo_origen_id: str
    tipo_bifurcacion: str
    trayectorias: List[TrayectoriaBifurcacion]
    paso_inicio: int
    pasos_exploracion: int = 10
    convergencias_detectadas: List[Dict] = None
    
    def __post_init__(self):
        if self.convergencias_detectadas is None:
            self.convergencias_detectadas = []

# Gestor principal de bifurcaciones TNFR
class BifurcationManagerTNFR:
    """Gestor canónico de bifurcaciones estructurales múltiples según principios TNFR"""
    
    def __init__(self):
        self.bifurcaciones_activas = {}  # {nodo_id: EspacioBifurcacion}
        self.trayectorias_exploradas = []
        self.convergencias_detectadas = []
        self.estadisticas_bifurcacion = {
            'total_bifurcaciones': 0,
            'bifurcaciones_simetricas': 0,
            'bifurcaciones_disonantes': 0,
            'bifurcaciones_fractales': 0,
            'convergencias_exitosas': 0,
            'trayectorias_colapsadas': 0
        }
    
    def detectar_bifurcacion_canonica(self, nodo, nodo_id, umbral_aceleracion=0.15):
        """Detecta si un nodo está en condiciones de bifurcación canónica TNFR"""
        try:
            # Métricas de aceleración estructural
            aceleracion = abs(nodo.get("d2EPI_dt2", 0))
            gradiente = abs(nodo.get("ΔNFR", 0))
            coherencia = nodo.get("Si", 0)
            energia = nodo.get("EPI", 0)
            frecuencia = nodo.get("νf", 1.0)
            
            # Validación de valores numéricos
            if not all(np.isfinite([aceleracion, gradiente, coherencia, energia, frecuencia])):
                return False, "valores_no_finitos"
            
            # Condiciones múltiples para bifurcación canónica
            condiciones = {
                'aceleracion_critica': aceleracion > umbral_aceleracion,
                'gradiente_alto': gradiente > 0.8,
                'coherencia_suficiente': coherencia > 0.4,
                'energia_minima': energia > 1.2,
                'frecuencia_activa': frecuencia > 0.8
            }
            
            # Evaluación de umbral de bifurcación
            condiciones_cumplidas = sum(condiciones.values())
            umbral_minimo = 3  # Al menos 3 de 5 condiciones
            
            # Determinación del tipo de bifurcación según las condiciones
            tipo_bifurcacion = self._determinar_tipo_bifurcacion(nodo, condiciones)
            
            es_bifurcacion = condiciones_cumplidas >= umbral_minimo
            return es_bifurcacion, tipo_bifurcacion
            
        except Exception as e:
            return False, "error_deteccion"
    
    def _determinar_tipo_bifurcacion(self, nodo, condiciones):
        """Determina el tipo de bifurcación según las condiciones estructurales"""
        aceleracion = abs(nodo.get("d2EPI_dt2", 0))
        coherencia = nodo.get("Si", 0)
        energia = nodo.get("EPI", 0)
        
        # Bifurcación simétrica: alta coherencia, aceleración moderada
        if coherencia > 0.7 and 0.15 < aceleracion < 0.3:
            return "simetrica"
        
        # Bifurcación disonante: baja coherencia, alta aceleración
        elif coherencia < 0.5 and aceleracion > 0.3:
            return "disonante"
        
        # Bifurcación fractal-expansiva: alta energía, alta aceleración
        elif energia > 2.0 and aceleracion > 0.25:
            return "fractal_expansiva"
        
        # Por defecto: simétrica
        else:
            return "simetrica"
    
    def generar_espacio_bifurcacion(self, nodo_id, nodo_data, tipo_bifurcacion, paso_actual):
        """Genera el espacio completo de bifurcación con múltiples trayectorias"""
        try:
            if tipo_bifurcacion == "simetrica":
                trayectorias = self._generar_bifurcacion_simetrica(nodo_id, nodo_data)
            elif tipo_bifurcacion == "disonante":
                trayectorias = self._generar_bifurcacion_disonante(nodo_id, nodo_data)
            elif tipo_bifurcacion == "fractal_expansiva":
                trayectorias = self._generar_bifurcacion_fractal_expansiva(nodo_id, nodo_data)
            else:
                trayectorias = self._generar_bifurcacion_simetrica(nodo_id, nodo_data)
            
            # Crear espacio de bifurcación
            espacio = EspacioBifurcacion(
                nodo_origen_id=nodo_id,
                tipo_bifurcacion=tipo_bifurcacion,
                trayectorias=trayectorias,
                paso_inicio=paso_actual,
                pasos_exploracion=random.randint(8, 15)  # Exploración variable
            )
            
            # Registrar estadísticas
            self.estadisticas_bifurcacion['total_bifurcaciones'] += 1
            self.estadisticas_bifurcacion[f'bifurcaciones_{tipo_bifurcacion}s'] += 1
            
            return espacio
            
        except Exception as e:
            return None
    
    def _generar_bifurcacion_simetrica(self, nodo_id, nodo_data):
        """Genera bifurcación simétrica con dos trayectorias complementarias"""
        trayectorias = []
        
        # Trayectoria A: Expansión coherente
        trayectoria_a = TrayectoriaBifurcacion(
            id=f"{nodo_id}_sym_A",
            tipo="expansion_coherente", 
            secuencia_glifica=["VA'L", "R'A", "I'L"],
            parametros_iniciales={
                "EPI": nodo_data.get("EPI", 1.0) * 1.2,
                "νf": nodo_data.get("νf", 1.0) * 1.1,
                "Si": nodo_data.get("Si", 0.5) * 1.15
            },
            convergencia_objetivo="coherencia_expandida"
        )
        
        # Trayectoria B: Contracción resonante
        trayectoria_b = TrayectoriaBifurcacion(
            id=f"{nodo_id}_sym_B",
            tipo="contraccion_resonante",
            secuencia_glifica=["NUL", "UM", "IL"],
            parametros_iniciales={
                "EPI": nodo_data.get("EPI", 1.0) * 0.8,
                "νf": nodo_data.get("νf", 1.0) * 0.9,
                "Si": nodo_data.get("Si", 0.5) * 1.2
            },
            convergencia_objetivo="coherencia_concentrada"
        )
        
        trayectorias.extend([trayectoria_a, trayectoria_b])
        return trayectorias
    
    def _generar_bifurcacion_disonante(self, nodo_id, nodo_data):
        """Genera bifurcación disonante con múltiples resoluciones"""
        trayectorias = []
        
        # Trayectoria A: Mutación directa
        trayectoria_a = TrayectoriaBifurcacion(
            id=f"{nodo_id}_dis_A",
            tipo="mutacion_directa", 
            secuencia_glifica=["THOL"],
            parametros_iniciales={
                "EPI": nodo_data.get("EPI", 1.0) * 1.5,
                "νf": nodo_data.get("νf", 1.0) * 1.3,
                "ΔNFR": nodo_data.get("ΔNFR", 0) * 1.4
            },
            convergencia_objetivo="mutacion_estable"
        )
        
        # Trayectoria B: Reorganización recursiva
        trayectoria_b = TrayectoriaBifurcacion(
            id=f"{nodo_id}_dis_B",
            tipo="reorganizacion_recursiva",
            secuencia_glifica=["RE'MESH", "NA'V"],
            parametros_iniciales={
                "EPI": nodo_data.get("EPI", 1.0) * 0.9,
                "νf": nodo_data.get("νf", 1.0),
                "Si": nodo_data.get("Si", 0.5) * 1.3
            },
            convergencia_objetivo="reorganizacion_estable"
        )
        
        # Trayectoria C: Silencio regenerativo
        trayectoria_c = TrayectoriaBifurcacion(
            id=f"{nodo_id}_dis_C",
            tipo="silencio_regenerativo",
            secuencia_glifica=["SH'A", "A'L"],
            parametros_iniciales={
                "EPI": nodo_data.get("EPI", 1.0) * 0.7,
                "νf": nodo_data.get("νf", 1.0) * 0.8,
                "Si": nodo_data.get("Si", 0.5) * 0.9
            },
            convergencia_objetivo="regeneracion_silenciosa"
        )
        
        trayectorias.extend([trayectoria_a, trayectoria_b, trayectoria_c])
        return trayectorias
    
    def _generar_bifurcacion_fractal_expansiva(self, nodo_id, nodo_data):
        """Genera bifurcación fractal-expansiva con sub-nodos derivados"""
        trayectorias = []
        
        # Trayectoria principal: Nodo padre con T'HOL
        trayectoria_padre = TrayectoriaBifurcacion(
            id=f"{nodo_id}_frac_padre",
            tipo="autoorganizacion_padre",
            secuencia_glifica=["THOL"],
            parametros_iniciales=nodo_data.copy(),
            convergencia_objetivo="autoorganizacion_estable"
        )
        
        # Sub-nodos derivados con variaciones
        for i in range(3):  # 3 sub-nodos derivados
            variacion = 0.8 + 0.4 * random.random()  # Variación 0.8-1.2
            
            trayectoria_derivado = TrayectoriaBifurcacion(
                id=f"{nodo_id}_frac_der_{i}",
                tipo="derivado_fractal",
                secuencia_glifica=["A'L", "E'N"],
                parametros_iniciales={
                    "EPI": nodo_data.get("EPI", 1.0) * variacion,
                    "νf": nodo_data.get("νf", 1.0) * variacion,
                    "Si": nodo_data.get("Si", 0.5) * (0.5 + 0.5 * variacion),
                    "derivado_de": nodo_id,
                    "factor_variacion": variacion
                },
                convergencia_objetivo="derivacion_coherente"
            )
            
            trayectorias.append(trayectoria_derivado)
        
        trayectorias.insert(0, trayectoria_padre)  # Padre al inicio
        return trayectorias

    def procesar_bifurcaciones_activas(self, G, paso_actual):
        """Procesa todas las bifurcaciones activas en el paso actual"""
        resultados = {
            'trayectorias_procesadas': 0,
            'convergencias_detectadas': 0,
            'bifurcaciones_completadas': [],
            'nuevos_nodos_generados': []
        }
        
        bifurcaciones_completadas = []
        
        for nodo_id, espacio_bifurcacion in list(self.bifurcaciones_activas.items()):
            try:
                # Verificar si la bifurcación ha completado su exploración
                pasos_transcurridos = paso_actual - espacio_bifurcacion.paso_inicio
                
                if pasos_transcurridos < espacio_bifurcacion.pasos_exploracion:
                    # Evolucionar trayectorias activas
                    resultado_evolucion = self._evolucionar_trayectorias(
                        espacio_bifurcacion, pasos_transcurridos, G
                    )
                    resultados['trayectorias_procesadas'] += resultado_evolucion['procesadas']
                    
                else:
                    # Convergencia final de trayectorias
                    resultado_convergencia = self._converger_bifurcacion(
                        espacio_bifurcacion, G, nodo_id
                    )
                    
                    if resultado_convergencia['exitosa']:
                        resultados['convergencias_detectadas'] += 1
                        resultados['bifurcaciones_completadas'].append(nodo_id)
                        resultados['nuevos_nodos_generados'].extend(
                            resultado_convergencia.get('nodos_generados', [])
                        )
                        bifurcaciones_completadas.append(nodo_id)
                        
                        # Actualizar estadísticas
                        self.estadisticas_bifurcacion['convergencias_exitosas'] += 1
                        
            except Exception as e:
                bifurcaciones_completadas.append(nodo_id)  # Eliminar bifurcación problemática
        
        # Limpiar bifurcaciones completadas
        for nodo_id in bifurcaciones_completadas:
            if nodo_id in self.bifurcaciones_activas:
                del self.bifurcaciones_activas[nodo_id]
        
        return resultados
    
    def _evolucionar_trayectorias(self, espacio_bifurcacion, paso_relativo, G):
        """Evoluciona las trayectorias de una bifurcación en el paso actual"""
        resultado = {'procesadas': 0, 'colapsadas': 0}
        
        for trayectoria in espacio_bifurcacion.trayectorias:
            if not trayectoria.activa:
                continue
                
            try:
                # Aplicar transformación glífica según el paso relativo
                if paso_relativo < len(trayectoria.secuencia_glifica):
                    glifo_actual = trayectoria.secuencia_glifica[paso_relativo]
                    
                    # Aplicar transformación específica de la trayectoria
                    self._aplicar_transformacion_trayectoria(
                        trayectoria, glifo_actual, G, espacio_bifurcacion.nodo_origen_id
                    )
                    
                    trayectoria.pasos_completados += 1
                    resultado['procesadas'] += 1
                
                # Evaluar viabilidad de la trayectoria
                viabilidad = self._evaluar_viabilidad_trayectoria(trayectoria)
                trayectoria.viabilidad = viabilidad
                
                # Marcar como inactiva si la viabilidad es muy baja
                if viabilidad < 0.2:
                    trayectoria.activa = False
                    resultado['colapsadas'] += 1
                    self.estadisticas_bifurcacion['trayectorias_colapsadas'] += 1
                    
            except Exception as e:
                trayectoria.activa = False
                resultado['colapsadas'] += 1
        
        return resultado
    
    def _aplicar_transformacion_trayectoria(self, trayectoria, glifo, G, nodo_origen_id):
        """Aplica una transformación glífica específica a una trayectoria"""
        try:
            # Obtener nodo origen desde el grafo
            if nodo_origen_id not in G.nodes():
                return
                
            nodo_data = G.nodes[nodo_origen_id]
            
            # Aplicar transformación según el glifo y tipo de trayectoria
            if glifo == "VAL":  # Expansión
                factor = 1.15 if trayectoria.tipo == "expansion_coherente" else 1.05
                trayectoria.parametros_iniciales["EPI"] *= factor
                
            elif glifo == "NUL":  # Contracción
                factor = 0.85 if trayectoria.tipo == "contraccion_resonante" else 0.95
                trayectoria.parametros_iniciales["EPI"] *= factor
                
            elif glifo == "ZHIR":  # Mutación
                trayectoria.parametros_iniciales["EPI"] += 0.5
                trayectoria.parametros_iniciales["νf"] *= 1.2
                
            elif glifo == "RA":  # Propagación
                trayectoria.parametros_iniciales["Si"] *= 1.1
                
            elif glifo == "IL":  # Estabilización
                # Convergencia hacia valores estables
                epi_objetivo = 1.5
                trayectoria.parametros_iniciales["EPI"] = (
                    trayectoria.parametros_iniciales["EPI"] * 0.8 + epi_objetivo * 0.2
                )
                
            elif glifo == "THOL":  # Autoorganización
                # Equilibrar todos los parámetros
                for param in ["EPI", "νf", "Si"]:
                    if param in trayectoria.parametros_iniciales:
                        valor_actual = trayectoria.parametros_iniciales[param]
                        valor_equilibrado = max(0.8, min(2.0, valor_actual))
                        trayectoria.parametros_iniciales[param] = valor_equilibrado    

        except Exception as e:
            pass            
    
    def _evaluar_viabilidad_trayectoria(self, trayectoria):
        """Evalúa la viabilidad estructural de una trayectoria"""
        try:
            # Obtener parámetros actuales
            epi = trayectoria.parametros_iniciales.get("EPI", 1.0)
            vf = trayectoria.parametros_iniciales.get("νf", 1.0)
            si = trayectoria.parametros_iniciales.get("Si", 0.5)
            
            # Validación numérica
            if not all(np.isfinite([epi, vf, si])):
                return 0.0
            
            # Criterios de viabilidad TNFR
            criterios = []
            
            # 1. Rango estructural válido
            criterios.append(1.0 if 0.3 <= epi <= 3.5 else 0.0)
            
            # 2. Frecuencia resonante
            criterios.append(1.0 if 0.2 <= vf <= 2.5 else 0.0)
            
            # 3. Coherencia mínima
            criterios.append(1.0 if si >= 0.1 else 0.0)
            
            # 4. Equilibrio energético
            ratio_equilibrio = min(epi/vf, vf/epi) if vf > 0 else 0
            criterios.append(ratio_equilibrio)
            
            # 5. Progreso en secuencia
            progreso = trayectoria.pasos_completados / max(len(trayectoria.secuencia_glifica), 1)
            criterios.append(min(progreso, 1.0))
            
            # Viabilidad como promedio ponderado
            viabilidad = np.mean(criterios)
            return max(0.0, min(1.0, viabilidad))
            
        except Exception as e:
            return 0.0
    
    def _converger_bifurcacion(self, espacio_bifurcacion, G, nodo_origen_id):
        """Convierte el espacio de bifurcación en configuración final estable"""
        resultado = {
            'exitosa': False,
            'nodos_generados': [],
            'configuracion_final': None
        }
        
        try:
            # Filtrar trayectorias viables
            trayectorias_viables = [
                t for t in espacio_bifurcacion.trayectorias 
                if t.activa and t.viabilidad > 0.3
            ]
            
            if not trayectorias_viables:
                return resultado
            
            # Ordenar por viabilidad
            trayectorias_viables.sort(key=lambda t: t.viabilidad, reverse=True)
            
            # Seleccionar trayectoria ganadora o fusionar múltiples
            if len(trayectorias_viables) == 1:
                # Una sola trayectoria viable
                resultado = self._aplicar_trayectoria_ganadora(
                    trayectorias_viables[0], G, nodo_origen_id
                )
            else:
                # Múltiples trayectorias: fusionar las más compatibles
                resultado = self._fusionar_trayectorias_compatibles(
                    trayectorias_viables[:3], G, nodo_origen_id  # Máximo 3 trayectorias
                )
            
            if resultado['exitosa']:
                # Registrar convergencia
                convergencia_info = {
                    'nodo_origen': nodo_origen_id,
                    'tipo_bifurcacion': espacio_bifurcacion.tipo_bifurcacion,
                    'trayectorias_fusionadas': len(trayectorias_viables),
                    'configuracion_final': resultado['configuracion_final']
                }
                self.convergencias_detectadas.append(convergencia_info)
            
            return resultado
            
        except Exception as e:
            return resultado
    
    def _aplicar_trayectoria_ganadora(self, trayectoria, G, nodo_origen_id):
        """Aplica la configuración de una trayectoria ganadora al nodo origen"""
        try:
            if nodo_origen_id not in G.nodes():
                return {'exitosa': False}
            
            nodo_data = G.nodes[nodo_origen_id]
            
            # Aplicar parámetros finales de la trayectoria
            for param, valor in trayectoria.parametros_iniciales.items():
                if param in ["EPI", "νf", "Si", "ΔNFR"] and np.isfinite(valor):
                    nodo_data[param] = max(0.1, min(3.0, valor))  # Límites de seguridad
            
            # Marcar convergencia exitosa
            nodo_data["ultima_bifurcacion"] = {
                'tipo': trayectoria.tipo,
                'convergencia': trayectoria.convergencia_objetivo,
                'viabilidad_final': trayectoria.viabilidad
            }
            
            return {
                'exitosa': True,
                'configuracion_final': trayectoria.parametros_iniciales.copy(),
                'nodos_generados': [nodo_origen_id]
            }
            
        except Exception as e:
            return {'exitosa': False}
    
    def _fusionar_trayectorias_compatibles(self, trayectorias, G, nodo_origen_id):
        """Fusiona múltiples trayectorias compatibles en una configuración híbrida"""
        try:
            if nodo_origen_id not in G.nodes():
                return {'exitosa': False}
            
            # Calcular promedios ponderados por viabilidad
            total_viabilidad = sum(t.viabilidad for t in trayectorias)
            if total_viabilidad == 0:
                return {'exitosa': False}
            
            configuracion_fusionada = {}
            
            for param in ["EPI", "νf", "Si", "ΔNFR"]:
                suma_ponderada = sum(
                    t.parametros_iniciales.get(param, 1.0) * t.viabilidad 
                    for t in trayectorias
                )
                valor_fusionado = suma_ponderada / total_viabilidad
                
                if np.isfinite(valor_fusionado):
                    configuracion_fusionada[param] = max(0.1, min(3.0, valor_fusionado))
            
            # Aplicar configuración fusionada
            nodo_data = G.nodes[nodo_origen_id]
            for param, valor in configuracion_fusionada.items():
                nodo_data[param] = valor
            
            # Marcar fusión exitosa
            nodo_data["ultima_bifurcacion"] = {
                'tipo': 'fusion_multiple',
                'trayectorias_fusionadas': len(trayectorias),
                'viabilidad_promedio': total_viabilidad / len(trayectorias)
            }
            
            return {
                'exitosa': True,
                'configuracion_final': configuracion_fusionada,
                'nodos_generados': [nodo_origen_id]
            }
            
        except Exception as e:
            return {'exitosa': False}
    
    def obtener_estadisticas_bifurcacion(self):
        """Retorna estadísticas completas del sistema de bifurcaciones"""
        stats = self.estadisticas_bifurcacion.copy()
        stats.update({
            'bifurcaciones_activas': len(self.bifurcaciones_activas),
            'tasa_convergencia': (
                stats['convergencias_exitosas'] / max(stats['total_bifurcaciones'], 1)
            ),
            'tasa_colapso': (
                stats['trayectorias_colapsadas'] / max(stats['total_bifurcaciones'] * 2.5, 1)
            )
        })
        return stats
    
# Funciones de integración específicas para OntoSim.py

def integrar_bifurcaciones_canonicas_en_simulacion(G, paso, coordinador_temporal, bifurcation_manager):
    """
    Función principal de integración de bifurcaciones canónicas en OntoSim
    Reemplaza la lógica simple de aplicar T'HOL automáticamente
    """
    resultados = {
        'nuevas_bifurcaciones': 0,
        'trayectorias_procesadas': 0,
        'convergencias_completadas': 0,
        'nodos_modificados': []
    }
    
    try:
        # Procesar bifurcaciones existentes primero
        if hasattr(bifurcation_manager, 'bifurcaciones_activas'):
            resultado_procesamiento = bifurcation_manager.procesar_bifurcaciones_activas(G, paso)
            resultados['trayectorias_procesadas'] = resultado_procesamiento['trayectorias_procesadas']
            resultados['convergencias_completadas'] = resultado_procesamiento['convergencias_detectadas']
            resultados['nodos_modificados'].extend(resultado_procesamiento['nuevos_nodos_generados'])
        
        # Detectar nuevas bifurcaciones
        for nodo_id, nodo_data in G.nodes(data=True):
            if nodo_data.get("estado") != "activo":
                continue
            
            # Verificar si el nodo ya está en bifurcación
            if nodo_id in bifurcation_manager.bifurcaciones_activas:
                continue
            
            # Detectar condición de bifurcación canónica
            es_bifurcacion, tipo_bifurcacion = bifurcation_manager.detectar_bifurcacion_canonica(
                nodo_data, nodo_id
            )
            
            if es_bifurcacion and tipo_bifurcacion != "error_deteccion":
                # Generar espacio de bifurcación múltiple
                espacio_bifurcacion = bifurcation_manager.generar_espacio_bifurcacion(
                    nodo_id, nodo_data, tipo_bifurcacion, paso
                )
                
                if espacio_bifurcacion:
                    # Registrar bifurcación activa
                    bifurcation_manager.bifurcaciones_activas[nodo_id] = espacio_bifurcacion
                    resultados['nuevas_bifurcaciones'] += 1
                    resultados['nodos_modificados'].append(nodo_id)                  
        
        return resultados
        
    except Exception as e:
        return resultados

def reemplazar_deteccion_bifurcacion_simple(G, paso, umbrales, bifurcation_manager):
    """
    Función que reemplaza la detección simple de bifurcaciones en OntoSim
    Debe llamarse en lugar del bloque original de detección de bifurcaciones
    """
    nodos_bifurcados = []
    
    try:
        # Parámetros dinámicos para detección
        umbral_aceleracion = umbrales.get('bifurcacion_umbral', 0.15)
        
        for nodo_id, nodo_data in G.nodes(data=True):
            if nodo_data.get("estado") != "activo":
                continue
            
            # Skip nodos ya en bifurcación
            if nodo_id in bifurcation_manager.bifurcaciones_activas:
                continue
            
            # Usar detección canónica en lugar de simple
            es_bifurcacion, tipo_bifurcacion = bifurcation_manager.detectar_bifurcacion_canonica(
                nodo_data, nodo_id, umbral_aceleracion
            )
            
            if es_bifurcacion:
                nodos_bifurcados.append((nodo_id, tipo_bifurcacion))
        
        return nodos_bifurcados
        
    except Exception as e:
        return []

# Función auxiliar para mostrar trayectorias activas detalladas
def mostrar_trayectorias_activas(bifurcation_manager):
    """Muestra detalles de las trayectorias activas"""
    if not bifurcation_manager.bifurcaciones_activas:
        return "No hay bifurcaciones activas"
    
    detalles = []
    for nodo_id, espacio in bifurcation_manager.bifurcaciones_activas.items():
        trayectorias_activas = [t for t in espacio.trayectorias if t.activa]
        viabilidades = [f"{t.viabilidad:.2f}" for t in trayectorias_activas]
        
        detalles.append(
            f"  {nodo_id}: {espacio.tipo_bifurcacion} "
            f"({len(trayectorias_activas)} activas, viabilidades: {viabilidades})"
        )
    
    return "Trayectorias activas:\n" + "\n".join(detalles)

# Función de limpieza para prevenir acumulación excesiva
def limpiar_bifurcaciones_obsoletas(bifurcation_manager, paso_actual, limite_pasos=50):
    """Limpia bifurcaciones que han excedido el tiempo máximo de exploración"""
    bifurcaciones_obsoletas = []
    
    for nodo_id, espacio in list(bifurcation_manager.bifurcaciones_activas.items()):
        pasos_transcurridos = paso_actual - espacio.paso_inicio
        
        if pasos_transcurridos > limite_pasos:
            bifurcaciones_obsoletas.append(nodo_id)
    
    for nodo_id in bifurcaciones_obsoletas:
        del bifurcation_manager.bifurcaciones_activas[nodo_id]
    
    return len(bifurcaciones_obsoletas)

# ------------------------- APLICACIÓN DE OPERADORES TNFR -------------------------

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

# ------------------------- EMERGENCIA -------------------------

def simular_emergencia(G, pasos=250):

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

    total_pasos = 250

    # Activación mínima inicial si todos están inactivos o silenciosos
    if all(G.nodes[n]["estado"] in ["latente", "silencio"] for n in G.nodes):
        for n in G.nodes:
            if G.nodes[n]["EPI"] > 0.8 and G.nodes[n]["νf"] > 0.5:
                G.nodes[n]["estado"] = "activo"
                G.nodes[n]["glifo"] = "AL"
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

        porcentaje = int((paso + 1) / total_pasos * 100)
        barra = "█" * (porcentaje // 2) + "-" * (50 - porcentaje // 2)
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

    with open("13_diagnostico_simbolico.json", "w", encoding="utf-8") as f:
        json.dump(diagnostico, f, indent=4, ensure_ascii=False)

    with open("11_registro_conexiones.json", "w", encoding="utf-8") as f:
        json.dump(registro_conexiones, f, indent=2, ensure_ascii=False)

    with open("12_historial_macronodos.json", "w", encoding="utf-8") as f:
        json.dump(historial_macronodos, f, indent=2, ensure_ascii=False)

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

def exportar_nodos_emitidos(G, emitidos_final=None, archivo="10_nodos_emitidos.csv"):
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
        
        # Crear archivo CSV con formato estándar
        with open(archivo, "w", encoding="utf-8") as f:
            # Cabecera CSV estándar TNFR
            f.write("nodo,glifo,EPI,Si,ΔNFR,θ,νf,categoria\n")
            
            # Exportar cada nodo emitido
            for n in emitidos_final:
                if n not in G.nodes:
                    continue  # Skip nodos que ya no existen
                
                nodo = G.nodes[n]
                
                # Validar y formatear valores
                epi = round(nodo.get("EPI", 1.0), 4)
                si = round(nodo.get("Si", 0.0), 4)
                dnfr = round(nodo.get("ΔNFR", 0.0), 4)
                theta = round(nodo.get("θ", 0.0), 4)
                vf = round(nodo.get("νf", 1.0), 4)
                glifo = nodo.get("glifo", "ninguno")
                categoria = nodo.get("categoria", "ninguna")
                
                # Escribir línea CSV
                f.write(f"{n},{glifo},{epi},{si},{dnfr},{theta},{vf},{categoria}\n")
        
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
