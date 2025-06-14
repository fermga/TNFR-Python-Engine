from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Optional
import networkx as nx
import numpy as np
import random

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

__all__ = [
    'TemporalCoordinatorTNFR',
    'EspacioBifurcacion',
    'BifurcationManagerTNFR',
    'calcular_umbrales_dinamicos',
    'aplicar_umbrales_dinamicos_conexiones',
    'integrar_tiempo_topologico_en_simulacion',
    'integrar_bifurcaciones_canonicas_en_simulacion',
    'reemplazar_deteccion_bifurcacion_simple',
    'mostrar_trayectorias_activas',
    'limpiar_bifurcaciones_obsoletas',
    'detectar_fase_simulacion',
    'inicializar_coordinador_temporal_canonico',
    'evaluar_condiciones_emergencia_dinamica',
    'evaluar_activacion_glifica_dinamica',
    'gestionar_conexiones_canonico'
]