"""
dynamics.py — TNFR refactor (θ como fase, glifos canónicos, umbrales coherentes)

CORRECCIÓN: Se han agregado funciones seguras para manejo de np.isfinite
que pueden fallar con datos de yfinance que vienen como strings, None, etc.

Ajustes clave respecto a la versión anterior:
- Docstrings y comentarios canónicos: **θ es FASE estructural**.
- Secuencias glíficas sin apóstrofos: "VAL", "RA", "IL", "REMESH", "NAV", "SHA", "AL", "EN".
- Corrección en estadísticas de bifurcación: conteo consistente (simétricas, disonantes, fractales).
- `evaluar_condiciones_emergencia_dinamica`: usa θ (fase) en vez de "fase" y no confunde θ_colapso con gradiente.
- `evaluar_activacion_glifica_dinamica`: el colapso por gradiente usa `bifurcacion_gradiente` (no θ_colapso).
- `aplicar_clamps_canonicos`: clamp de θ a [0,1] además de EPI/Si/νf/ΔNFR.
- Imports ampliados desde constants con fallbacks seguros.
- **NUEVO**: Funciones seguras para manejo de tipos de datos problemáticos
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Optional, Any
import networkx as nx
import numpy as np
import random
from tqdm import tqdm

# =========================================================================================
# FUNCIONES HELPER SEGURAS (CORRECCIÓN DEL ERROR np.isfinite)
# =========================================================================================

def _safe_isfinite(value):
    """Versión segura de np.isfinite que maneja cualquier tipo de dato"""
    try:
        # Intentar convertir a float primero
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return False
        elif value is None:
            return False
        elif hasattr(value, 'isna') and callable(value.isna):
            try:
                if value.isna():  # pandas NaT/NaN
                    return False
            except (ValueError, TypeError):
                pass
        elif hasattr(value, '__array__') and hasattr(value, 'dtype'):
            # Para arrays numpy/pandas
            try:
                return bool(np.isfinite(value))
            except:
                return False
        
        return bool(np.isfinite(value))
    except (ValueError, TypeError, AttributeError):
        return False

def _safe_float(value, default=0.0):
    """Convierte de manera segura un valor a float"""
    try:
        if isinstance(value, str):
            return float(value)
        elif value is None:
            return default
        elif hasattr(value, 'isna') and callable(value.isna):
            if value.isna():
                return default
        
        result = float(value)
        return result if np.isfinite(result) else default
    except (ValueError, TypeError, AttributeError):
        return default

# =========================================================================================
# CONSTANTES / FALLBACKS (centraliza "magic numbers" si constants.py no está disponible)
# =========================================================================================
try:
    from constants import (
        EPI_MAX_GLOBAL,
        SI_MIN, SI_MAX,
        VF_MIN, VF_MAX,
        DELTA_NFR_MAX_ABS,
        EPS_EPI_STABLE, EPS_DNFR_STABLE, EPS_DERIV_STABLE, EPS_ACCEL_STABLE,
    )
except Exception:  # fallbacks seguros
    EPI_MAX_GLOBAL = 3.5
    SI_MIN, SI_MAX = 0.0, 1.0
    VF_MIN, VF_MAX = 0.1, 5.0
    DELTA_NFR_MAX_ABS = 5.0
    EPS_EPI_STABLE = 0.01
    EPS_DNFR_STABLE = 0.05
    EPS_DERIV_STABLE = 0.01
    EPS_ACCEL_STABLE = 0.01

# =========================================================================================
# HELPERS CANÓNICOS (CORREGIDOS)
# =========================================================================================

def _asegurar_defaults_nodo(nodo: Dict[str, Any]):
    """Garantiza llaves mínimas para operar sin KeyError (canónico defensivo).
    
    CORRECCIÓN: Usa _safe_isfinite y _safe_float para evitar errores con
    datos de yfinance que pueden venir como strings, None, pandas NaT, etc.
    """
    defaults = {
        "EPI": 1.0,
        "Si": 0.5,
        "νf": 1.0,
        "ΔNFR": 0.0,
        "θ": 0.5,  # θ = fase estructural
        "estado": "latente",
    }
    for k, v in defaults.items():
        if k not in nodo or not _safe_isfinite(nodo.get(k, v)):
            nodo[k] = _safe_float(nodo.get(k, v), v)

def medir_C(G: nx.Graph) -> float:
    """Coherencia total C(t) ≈ ⟨EPI⟩."""
    if len(G) == 0:
        return 0.0
    EPIs = [_safe_float(G.nodes[n].get("EPI", 1.0), 1.0) for n in G.nodes]
    return float(np.mean(EPIs))

# =========================================================================================
# ECUACIÓN NODAL TNFR (∂EPI/∂t = νf · ΔNFR)
# =========================================================================================

def actualizar_EPI_por_ecuacion_nodal(G: nx.Graph, dt: float = 1.0) -> None:
    """
    Integra la ecuación nodal: ∂EPI/∂t = νf · ΔNFR.
    - Actualiza EPI
    - Mantiene historial corto EPI_prev*, dEPI_dt, d2EPI_dt2 (útil para T'HOL/bifurcación)
    
    CORRECCIÓN: Usa _safe_float para conversiones seguras.
    """
    for n in G.nodes:
        nodo = G.nodes[n]
        _asegurar_defaults_nodo(nodo)

        # Derivada primera por ecuación nodal
        dEPI_dt = _safe_float(nodo.get("νf", 1.0), 1.0) * _safe_float(nodo.get("ΔNFR", 0.0), 0.0)
        prev_dEPI_dt = _safe_float(nodo.get("dEPI_dt", 0.0), 0.0)

        # Historial EPI
        epi_actual = _safe_float(nodo.get("EPI", 1.0), 1.0)
        nodo["EPI_prev3"] = _safe_float(nodo.get("EPI_prev2", epi_actual), epi_actual)
        nodo["EPI_prev2"] = _safe_float(nodo.get("EPI_prev", epi_actual), epi_actual)
        nodo["EPI_prev"] = epi_actual

        # Integración explícita (Euler) con saneamiento numérico
        EPI_new = epi_actual + dt * dEPI_dt
        if not _safe_isfinite(EPI_new):
            EPI_new = epi_actual  # no avanzar ante NaN/inf
        nodo["EPI"] = EPI_new

        # Derivadas almacenadas
        nodo["dEPI_dt_prev"] = prev_dEPI_dt
        nodo["dEPI_dt"] = dEPI_dt
        nodo["d2EPI_dt2"] = (dEPI_dt - prev_dEPI_dt) / max(1e-6, dt)

# =========================================================================================
# CLAMPS CANÓNICOS (sanity pass post-tick) - CORREGIDOS
# =========================================================================================

def aplicar_clamps_canonicos(G: nx.Graph, umbrales: Optional[Dict[str, float]] = None) -> None:
    """Ajustes suaves y límites globales para mantener coherencia numérica.
    - EPI ≥ 0 y ≤ EPI_max (dinámico si se provee)
    - 0 ≤ Si ≤ 1
    - VF_MIN ≤ νf ≤ VF_MAX
    - |ΔNFR| ≤ max_abs_ΔNFR (umbrales o fallback)
    - **θ clamp a [0,1]** (fase estructural)
    
    CORRECCIÓN: Usa _safe_float para todas las conversiones.
    """
    EPI_max = _safe_float(
        (umbrales or {}).get("EPI_max_dinamico", EPI_MAX_GLOBAL), 
        EPI_MAX_GLOBAL
    )
    max_abs_delta = _safe_float(
        (umbrales or {}).get("bifurcacion_gradiente", DELTA_NFR_MAX_ABS), 
        DELTA_NFR_MAX_ABS
    )

    for n in G.nodes:
        nodo = G.nodes[n]
        _asegurar_defaults_nodo(nodo)

        # Sanitizar y acotar usando conversiones seguras
        epi = _safe_float(nodo.get("EPI", 1.0), 1.0)
        si = _safe_float(nodo.get("Si", 0.5), 0.5)
        vf = _safe_float(nodo.get("νf", 1.0), 1.0)
        dln = _safe_float(nodo.get("ΔNFR", 0.0), 0.0)
        theta = _safe_float(nodo.get("θ", 0.5), 0.5)

        # Aplicar límites
        epi = min(EPI_max, max(0.0, epi))
        si = min(SI_MAX, max(SI_MIN, si))
        vf = min(VF_MAX, max(VF_MIN, max(0.001, vf)))  # Evitar νf = 0
        dln = max(-max_abs_delta, min(max_abs_delta, dln))
        theta = max(0.0, min(1.0, theta))

        nodo["EPI"], nodo["Si"], nodo["νf"], nodo["ΔNFR"], nodo["θ"] = epi, si, vf, dln, theta

# =========================================================================================
# GESTIÓN TEMPORAL TOPOLÓGICA TNFR (mantenido igual pero con correcciones seguras)
# =========================================================================================

class TemporalCoordinatorTNFR:
    """
    Coordinador temporal canónico que gestiona tiempo topológico variable
    según frecuencias estructurales νf de cada NFR y principios de entrainment.
    """

    def __init__(self, sincronizacion_global: bool = True, pulsos_reorganizacion: int = 50):
        # Configuración temporal canónica
        self.sincronizacion_global = sincronizacion_global
        self.frecuencia_pulsos = pulsos_reorganizacion
        self.tiempo_topologico = 0.0

        # Estados temporales de nodos
        self.cronometros_nodales: Dict[Any, float] = {}
        self.fases_sincronizacion: Dict[Any, float] = {}
        self.ultimas_activaciones: Dict[Any, float] = {}

        # Historial de sincronización
        self.historial_entrainment: List[Dict[str, Any]] = []
        self.historial_coherencia_temporal: List[tuple[int, float]] = []

        # Cola de eventos temporales
        self.cola_eventos: List[tuple[float, Any, str]] = []  # [(tiempo_activacion, nodo_id, tipo_evento)]

        # Parámetros de resonancia temporal
        self.umbral_resonancia = 0.15  # diferencia máxima en νf para resonancia
        self.factor_aceleracion = 1.8  # aceleración temporal por coherencia alta

    def calcular_paso_temporal_nodal(self, nodo: Dict[str, Any], paso_global: int) -> float:
        # Validación preventiva de parámetros usando funciones seguras
        vf_nodo = _safe_float(nodo.get("νf", 1.0), 1.0)
        if vf_nodo <= 0:
            vf_nodo = 1.0
            nodo["νf"] = 1.0  # Corrección in-situ

        Si_nodo = _safe_float(nodo.get("Si", 0.5), 0.5)
        theta_nodo = _safe_float(nodo.get("θ", 0.5), 0.5)
        estado = nodo.get("estado", "latente")

        # Paso base según frecuencia estructural (inversa de νf)
        # Alta frecuencia = pasos más pequeños (más actividad)
        paso_base = 1.0 / max(0.1, vf_nodo)

        # Factor de coherencia: mayor Si permite pasos más largos (estabilidad)
        factor_coherencia = 0.5 + 0.5 * Si_nodo

        # Factor de activación: nodos activos necesitan pasos más pequeños
        factor_activacion = {
            "activo": 0.7,   # pasos más pequeños para actividad
            "latente": 1.0,  # pasos normales
            "silencio": 1.5, # pasos más grandes en silencio
            "inactivo": 2.0, # pasos muy grandes si inactivo
        }.get(estado, 1.0)

        # Factor de umbral: cerca de bifurcación (fase alta) = pasos más pequeños
        factor_umbral = 1.0 - 0.3 * min(1.0, theta_nodo)

        # Combinar todos los factores
        paso_temporal = paso_base * factor_coherencia * factor_activacion * factor_umbral

        # Limitar al rango [0.1, 5.0] para evitar extremos
        return max(0.1, min(5.0, paso_temporal))

    def detectar_nodos_resonantes(self, G: nx.Graph):
        """Detecta grupos de nodos con frecuencias νf compatibles para entrainment."""
        nodos_por_frecuencia = defaultdict(list)

        # Agrupar nodos por bandas de frecuencia
        for nodo_id, nodo_data in G.nodes(data=True):
            vf = _safe_float(nodo_data.get("νf", 1.0), 1.0)
            if vf > 0 and _safe_isfinite(self.umbral_resonancia):
                try:
                    ratio = vf / self.umbral_resonancia
                    if _safe_isfinite(ratio) and abs(ratio) < 1e6:  # Límite de seguridad
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
                coherencias = [_safe_float(G.nodes[n].get("Si", 0), 0) for n in nodos]
                if np.mean(coherencias) > 0.4:  # grupo coherente
                    grupos_resonantes.append(
                        {
                            "banda_frecuencia": banda,
                            "nodos": nodos,
                            "coherencia_grupal": float(np.mean(coherencias)),
                            "tamaño": len(nodos),
                        }
                    )

        return grupos_resonantes

    def sincronizar_grupo_resonante(self, G: nx.Graph, grupo: Dict[str, Any]) -> int:
        """Sincroniza temporalmente un grupo de nodos resonantes mediante entrainment."""
        nodos = grupo["nodos"]

        # Calcular fase de sincronización grupal
        fases_actuales = [self.fases_sincronizacion.get(n, 0.0) for n in nodos]
        fase_promedio = float(np.mean(fases_actuales)) if fases_actuales else 0.0

        # Factor de atracción hacia sincronización
        for nodo_id in nodos:
            fase_actual = self.fases_sincronizacion.get(nodo_id, 0.0)

            # Corrección de fase hacia el promedio grupal
            diferencia_fase = fase_promedio - fase_actual
            factor_correccion = 0.1 * _safe_float(G.nodes[nodo_id].get("Si", 0.5), 0.5)  # más Si = más atraído

            # Aplicar corrección suave
            nueva_fase = fase_actual + factor_correccion * diferencia_fase
            self.fases_sincronizacion[nodo_id] = nueva_fase % (2 * np.pi)

            # Ajustar cronómetro nodal para sincronización
            ajuste_temporal = np.sin(diferencia_fase) * 0.05
            cronometro_actual = self.cronometros_nodales.get(nodo_id, 0.0)
            self.cronometros_nodales[nodo_id] = cronometro_actual + ajuste_temporal

        return len(nodos)  # cantidad de nodos sincronizados

    def generar_pulso_reorganizacion_global(self, G: nx.Graph, paso_global: int) -> bool:
        """Genera pulso de reorganización global que sincroniza toda la red."""
        if paso_global % self.frecuencia_pulsos != 0:
            return False

        # Calcular coherencia global actual
        EPIs = [_safe_float(G.nodes[n].get("EPI", 1.0), 1.0) for n in G.nodes]
        coherencia_global = float(np.mean(EPIs)) if EPIs else 0.0

        # Intensidad del pulso basada en necesidad de reorganización
        variabilidad_EPI = float(np.std(EPIs)) if EPIs else 0.0
        intensidad_pulso = min(1.0, variabilidad_EPI / max(1e-6, coherencia_global)) if coherencia_global > 0 else 0.0

        # Aplicar pulso a todos los nodos
        for nodo_id in G.nodes:
            # Reset parcial del cronómetro según intensidad
            cronometro_actual = self.cronometros_nodales.get(nodo_id, 0.0)
            ajuste = intensidad_pulso * 0.2 * random.uniform(-1, 1)
            self.cronometros_nodales[nodo_id] = cronometro_actual + ajuste

            # Sincronizar fases hacia coherencia global en función de νf
            fase_actual = self.fases_sincronizacion.get(nodo_id, 0.0)
            vf = _safe_float(G.nodes[nodo_id].get("νf", 1.0), 1.0)
            tiempo_normalizado = self.tiempo_topologico % (4 * np.pi)
            fase_objetivo = (vf * tiempo_normalizado) % (2 * np.pi) if _safe_isfinite(vf) else 0.0
            diferencia = fase_objetivo - fase_actual
            self.fases_sincronizacion[nodo_id] = fase_actual + 0.3 * diferencia

        return True

    def calcular_simultaneidad_eventos(self, G: nx.Graph, eventos_candidatos: List[tuple]) -> List[List[tuple]]:
        """Determina qué eventos pueden ocurrir simultáneamente basado en coherencia."""
        if len(eventos_candidatos) <= 1:
            return [eventos_candidatos] if eventos_candidatos else []

        eventos_simultaneos: List[List[tuple]] = []
        eventos_procesados: set = set()

        for i, (tiempo_i, nodo_i, evento_i) in enumerate(eventos_candidatos):
            if i in eventos_procesados:
                continue

            grupo_simultaneo = [(tiempo_i, nodo_i, evento_i)]
            eventos_procesados.add(i)

            # Buscar eventos compatibles para simultaneidad
            for j, (tiempo_j, nodo_j, evento_j) in enumerate(eventos_candidatos[i + 1 :], i + 1):
                if j in eventos_procesados:
                    continue

                # Criterios: cercanía temporal + compatibilidad estructural
                if abs(tiempo_i - tiempo_j) > 0.1:
                    continue

                nodo_data_i = G.nodes[nodo_i]
                nodo_data_j = G.nodes[nodo_j]

                diferencia_vf = abs(_safe_float(nodo_data_i.get("νf", 1), 1) - _safe_float(nodo_data_j.get("νf", 1), 1))
                diferencia_Si = abs(_safe_float(nodo_data_i.get("Si", 0), 0) - _safe_float(nodo_data_j.get("Si", 0), 0))

                if (
                    diferencia_vf < self.umbral_resonancia
                    and diferencia_Si < 0.3
                    and len(grupo_simultaneo) < 5
                ):
                    grupo_simultaneo.append((tiempo_j, nodo_j, evento_j))
                    eventos_procesados.add(j)

            eventos_simultaneos.append(grupo_simultaneo)

        return eventos_simultaneos

    def avanzar_tiempo_topologico(self, G: nx.Graph, paso_global: int) -> Dict[str, Any]:
        """Función principal que avanza el tiempo topológico de la red."""
        eventos_este_paso: List[tuple] = []
        grupos_resonantes = self.detectar_nodos_resonantes(G)

        if self.tiempo_topologico > 1e6 or not _safe_isfinite(self.tiempo_topologico):
            self.tiempo_topologico = self.tiempo_topologico % (8 * np.pi)  # Normalizar
        if not _safe_isfinite(self.tiempo_topologico):
            self.tiempo_topologico = 0.0

        # Procesar cada nodo con su tiempo topológico individual
        for nodo_id, nodo_data in G.nodes(data=True):
            # Inicializar cronómetro si es necesario
            if nodo_id not in self.cronometros_nodales:
                self.cronometros_nodales[nodo_id] = 0.0
                self.fases_sincronizacion[nodo_id] = random.uniform(0, 2 * np.pi)

            # Calcular paso temporal para este nodo
            paso_nodal = self.calcular_paso_temporal_nodal(nodo_data, paso_global)

            # Avanzar cronómetro nodal
            self.cronometros_nodales[nodo_id] += paso_nodal

            # Actualizar fase de sincronización
            vf = _safe_float(nodo_data.get("νf", 1.0), 1.0)
            incremento_fase = 2 * np.pi * paso_nodal * vf
            self.fases_sincronizacion[nodo_id] = (
                self.fases_sincronizacion[nodo_id] + incremento_fase
            ) % (2 * np.pi)

            # Verificar si el nodo debe activarse en este paso (período ≈ 1/νf)
            tiempo_desde_activacion = self.cronometros_nodales[nodo_id] - self.ultimas_activaciones.get(nodo_id, 0)
            umbral_activacion = 1.0 / max(0.1, vf)

            if tiempo_desde_activacion >= umbral_activacion:
                eventos_este_paso.append((self.cronometros_nodales[nodo_id], nodo_id, "activacion_temporal"))
                self.ultimas_activaciones[nodo_id] = self.cronometros_nodales[nodo_id]

        # Control de desbordamiento de cronómetros
        for nodo_id in list(self.cronometros_nodales.keys()):
            if self.cronometros_nodales[nodo_id] > 1e4:
                self.cronometros_nodales[nodo_id] = self.cronometros_nodales[nodo_id] % 100.0

        # Sincronizar grupos resonantes y pulso global
        nodos_sincronizados = sum(self.sincronizar_grupo_resonante(G, g) for g in grupos_resonantes)
        pulso_global = self.generar_pulso_reorganizacion_global(G, paso_global)

        # Calcular eventos simultáneos
        grupos_simultaneos = self.calcular_simultaneidad_eventos(G, eventos_este_paso)

        # Avanzar tiempo topológico global
        incremento_global = (
            np.mean([self.calcular_paso_temporal_nodal(G.nodes[n], paso_global) for n in G.nodes])
            if len(G.nodes) > 0 else 0.0
        )
        self.tiempo_topologico += incremento_global

        # Registrar estadísticas temporales
        coherencia_temporal = self.calcular_coherencia_temporal(G)
        self.historial_coherencia_temporal.append((paso_global, coherencia_temporal))

        # Registrar información de entrainment
        self.historial_entrainment.append({
            "paso": paso_global,
            "grupos_resonantes": len(grupos_resonantes),
            "nodos_sincronizados": nodos_sincronizados,
            "eventos_simultaneos": len([g for g in grupos_simultaneos if len(g) > 1]),
            "pulso_global": pulso_global,
            "coherencia_temporal": coherencia_temporal,
        })

        return {
            "tiempo_topologico": self.tiempo_topologico,
            "grupos_resonantes": grupos_resonantes,
            "eventos_simultaneos": grupos_simultaneos,
            "estadisticas": self.historial_entrainment[-1],
        }

    def calcular_coherencia_temporal(self, G: nx.Graph) -> float:
        """Calcula la coherencia temporal global de la red (Kuramoto + cronómetros)."""
        if len(G.nodes) == 0:
            return 0.0

        fases = [self.fases_sincronizacion.get(n, 0.0) for n in G.nodes]
        suma_compleja = sum(np.exp(1j * fase) for fase in fases)
        parametro_orden = abs(suma_compleja) / len(fases)

        cronometros = [self.cronometros_nodales.get(n, 0.0) for n in G.nodes]
        variabilidad_cronometros = np.std(cronometros) / (np.mean(cronometros) + 0.1)
        coherencia_cronometros = 1.0 / (1.0 + variabilidad_cronometros)

        return float(0.6 * parametro_orden + 0.4 * coherencia_cronometros)


def inicializar_coordinador_temporal_canonico():
    """Inicializa el coordinador temporal canónico para OntoSim."""
    return TemporalCoordinatorTNFR(sincronizacion_global=True, pulsos_reorganizacion=75)


def integrar_tiempo_topologico_en_simulacion(G: nx.Graph, paso: int, coordinador_temporal: TemporalCoordinatorTNFR):
    """Integración por paso (llamar desde `simular_emergencia`).
    - Avanza tiempo topológico y modula νf por fase (θ)
    - Integra ecuación nodal TNFR y aplica clamps canónicos
    """
    resultado_temporal = coordinador_temporal.avanzar_tiempo_topologico(G, paso)

    # Efectos temporales a nodos (retroalimentación en νf)
    for nodo_id, nodo_data in G.nodes(data=True):
        fase = coordinador_temporal.fases_sincronizacion.get(nodo_id, 0.0)
        modulacion_temporal = 1.0 + 0.1 * np.sin(fase)  # modulación suave
        vf_actual = _safe_float(nodo_data.get("νf", 1.0), 1.0)
        nodo_data["νf"] = vf_actual * modulacion_temporal
        # Persistencia temporal mínima
        nodo_data["cronometro_topologico"] = coordinador_temporal.cronometros_nodales.get(nodo_id, 0.0)
        nodo_data["fase_temporal"] = fase
        nodo_data["ultima_sincronizacion"] = paso

    # Integración nodal y clamps
    actualizar_EPI_por_ecuacion_nodal(G, dt=1.0)
    aplicar_clamps_canonicos(G)

    return resultado_temporal


# =========================================================================================
# Sistema de Bifurcaciones Estructurales Múltiples (mantenido igual)
# =========================================================================================

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


@dataclass
class EspacioBifurcacion:
    """Espacio completo de una bifurcación con múltiples trayectorias"""
    nodo_origen_id: str
    tipo_bifurcacion: str
    trayectorias: List[TrayectoriaBifurcacion]
    paso_inicio: int
    pasos_exploracion: int = 10
    convergencias_detectadas: List[Dict[str, Any]] | None = None

    def __post_init__(self):
        if self.convergencias_detectadas is None:
            self.convergencias_detectadas = []


class BifurcationManagerTNFR:
    """Gestor canónico de bifurcaciones estructurales múltiples según principios TNFR"""

    def __init__(self):
        self.bifurcaciones_activas: Dict[Any, EspacioBifurcacion] = {}
        self.trayectorias_exploradas: List[TrayectoriaBifurcacion] = []
        self.convergencias_detectadas: List[Dict[str, Any]] = []
        self.estadisticas_bifurcacion = {
            "total_bifurcaciones": 0,
            "bifurcaciones_simetricas": 0,
            "bifurcaciones_disonantes": 0,
            "bifurcaciones_fractales": 0,
            "convergencias_exitosas": 0,
            "trayectorias_colapsadas": 0,
        }

    def detectar_bifurcacion_canonica(self, nodo: Dict[str, Any], nodo_id: Any, umbral_aceleracion: float = 0.15):
        """Detecta si un nodo está en condiciones de bifurcación canónica TNFR"""
        try:
            aceleracion = abs(_safe_float(nodo.get("d2EPI_dt2", 0), 0))
            gradiente = abs(_safe_float(nodo.get("ΔNFR", 0), 0))
            coherencia = _safe_float(nodo.get("Si", 0), 0)
            energia = _safe_float(nodo.get("EPI", 0), 0)
            frecuencia = _safe_float(nodo.get("νf", 1.0), 1.0)

            if not all(_safe_isfinite(x) for x in [aceleracion, gradiente, coherencia, energia, frecuencia]):
                return False, "valores_no_finitos"

            condiciones = {
                "aceleracion_critica": aceleracion > umbral_aceleracion,
                "gradiente_alto": gradiente > 0.8,
                "coherencia_suficiente": coherencia > 0.4,
                "energia_minima": energia > 1.2,
                "frecuencia_activa": frecuencia > 0.8,
            }

            condiciones_cumplidas = sum(condiciones.values())
            tipo_bifurcacion = self._determinar_tipo_bifurcacion(nodo, condiciones)
            es_bifurcacion = condiciones_cumplidas >= 3  # 3 de 5
            return es_bifurcacion, tipo_bifurcacion

        except Exception:
            return False, "error_deteccion"

    def _determinar_tipo_bifurcacion(self, nodo: Dict[str, Any], condiciones: Dict[str, bool]) -> str:
        aceleracion = abs(_safe_float(nodo.get("d2EPI_dt2", 0), 0))
        coherencia = _safe_float(nodo.get("Si", 0), 0)
        energia = _safe_float(nodo.get("EPI", 0), 0)

        if coherencia > 0.7 and 0.15 < aceleracion < 0.3:
            return "simetrica"
        elif coherencia < 0.5 and aceleracion > 0.3:
            return "disonante"
        elif energia > 2.0 and aceleracion > 0.25:
            return "fractal_expansiva"
        else:
            return "simetrica"

    def generar_espacio_bifurcacion(self, nodo_id: Any, nodo_data: Dict[str, Any], tipo_bifurcacion: str, paso_actual: int) -> Optional[EspacioBifurcacion]:
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

            espacio = EspacioBifurcacion(
                nodo_origen_id=nodo_id,
                tipo_bifurcacion=tipo_bifurcacion,
                trayectorias=trayectorias,
                paso_inicio=paso_actual,
                pasos_exploracion=random.randint(8, 15),
            )

            # Estadísticas coherentes con las claves declaradas
            self.estadisticas_bifurcacion["total_bifurcaciones"] += 1
            if tipo_bifurcacion == "simetrica":
                self.estadisticas_bifurcacion["bifurcaciones_simetricas"] += 1
            elif tipo_bifurcacion == "disonante":
                self.estadisticas_bifurcacion["bifurcaciones_disonantes"] += 1
            elif tipo_bifurcacion == "fractal_expansiva":
                self.estadisticas_bifurcacion["bifurcaciones_fractales"] += 1

            return espacio

        except Exception:
            return None

    def _generar_bifurcacion_simetrica(self, nodo_id: Any, nodo_data: Dict[str, Any]) -> List[TrayectoriaBifurcacion]:
        """Genera bifurcación simétrica con dos trayectorias complementarias"""
        trayectorias: List[TrayectoriaBifurcacion] = []

        trayectoria_a = TrayectoriaBifurcacion(
            id=f"{nodo_id}_sym_A",
            tipo="expansion_coherente",
            secuencia_glifica=["VAL", "RA", "IL"],
            parametros_iniciales={
                "EPI": _safe_float(nodo_data.get("EPI", 1.0), 1.0) * 1.2,
                "νf": _safe_float(nodo_data.get("νf", 1.0), 1.0) * 1.1,
                "Si": _safe_float(nodo_data.get("Si", 0.5), 0.5) * 1.15,
            },
            convergencia_objetivo="coherencia_expandida",
        )

        trayectoria_b = TrayectoriaBifurcacion(
            id=f"{nodo_id}_sym_B",
            tipo="contraccion_resonante",
            secuencia_glifica=["NUL", "UM", "IL"],
            parametros_iniciales={
                "EPI": _safe_float(nodo_data.get("EPI", 1.0), 1.0) * 0.8,
                "νf": _safe_float(nodo_data.get("νf", 1.0), 1.0) * 0.9,
                "Si": _safe_float(nodo_data.get("Si", 0.5), 0.5) * 1.2,
            },
            convergencia_objetivo="coherencia_concentrada",
        )

        trayectorias.extend([trayectoria_a, trayectoria_b])
        return trayectorias

    def _generar_bifurcacion_disonante(self, nodo_id: Any, nodo_data: Dict[str, Any]) -> List[TrayectoriaBifurcacion]:
        """Genera bifurcación disonante con múltiples resoluciones"""
        trayectorias: List[TrayectoriaBifurcacion] = []

        trayectoria_a = TrayectoriaBifurcacion(
            id=f"{nodo_id}_dis_A",
            tipo="mutacion_directa",
            secuencia_glifica=["THOL"],
            parametros_iniciales={
                "EPI": _safe_float(nodo_data.get("EPI", 1.0), 1.0) * 1.5,
                "νf": _safe_float(nodo_data.get("νf", 1.0), 1.0) * 1.3,
                "ΔNFR": _safe_float(nodo_data.get("ΔNFR", 0), 0) * 1.4,
            },
            convergencia_objetivo="mutacion_estable",
        )

        trayectoria_b = TrayectoriaBifurcacion(
            id=f"{nodo_id}_dis_B",
            tipo="reorganizacion_recursiva",
            secuencia_glifica=["REMESH", "NAV"],
            parametros_iniciales={
                "EPI": _safe_float(nodo_data.get("EPI", 1.0), 1.0) * 0.9,
                "νf": _safe_float(nodo_data.get("νf", 1.0), 1.0),
                "Si": _safe_float(nodo_data.get("Si", 0.5), 0.5) * 1.3,
            },
            convergencia_objetivo="reorganizacion_estable",
        )

        trayectoria_c = TrayectoriaBifurcacion(
            id=f"{nodo_id}_dis_C",
            tipo="silencio_regenerativo",
            secuencia_glifica=["SHA", "AL"],
            parametros_iniciales={
                "EPI": _safe_float(nodo_data.get("EPI", 1.0), 1.0) * 0.7,
                "νf": _safe_float(nodo_data.get("νf", 1.0), 1.0) * 0.8,
                "Si": _safe_float(nodo_data.get("Si", 0.5), 0.5) * 0.9,
            },
            convergencia_objetivo="regeneracion_silenciosa",
        )

        trayectorias.extend([trayectoria_a, trayectoria_b, trayectoria_c])
        return trayectorias

    def _generar_bifurcacion_fractal_expansiva(self, nodo_id: Any, nodo_data: Dict[str, Any]) -> List[TrayectoriaBifurcacion]:
        """Genera bifurcación fractal-expansiva con sub-nodos derivados"""
        trayectorias: List[TrayectoriaBifurcacion] = []

        # Convertir nodo_data a diccionario de floats seguros
        nodo_data_safe = {k: _safe_float(v, 1.0 if k in ["EPI", "νf"] else 0.5) for k, v in nodo_data.items()}

        trayectoria_padre = TrayectoriaBifurcacion(
            id=f"{nodo_id}_frac_padre",
            tipo="autoorganizacion_padre",
            secuencia_glifica=["THOL"],
            parametros_iniciales=nodo_data_safe.copy(),
            convergencia_objetivo="autoorganizacion_estable",
        )

        for i in range(3):  # 3 sub-nodos derivados
            variacion = 0.8 + 0.4 * random.random()  # 0.8–1.2
            trayectoria_derivado = TrayectoriaBifurcacion(
                id=f"{nodo_id}_frac_der_{i}",
                tipo="derivado_fractal",
                secuencia_glifica=["AL", "EN"],
                parametros_iniciales={
                    "EPI": _safe_float(nodo_data.get("EPI", 1.0), 1.0) * variacion,
                    "νf": _safe_float(nodo_data.get("νf", 1.0), 1.0) * variacion,
                    "Si": _safe_float(nodo_data.get("Si", 0.5), 0.5) * (0.5 + 0.5 * variacion),
                    "derivado_de": str(nodo_id),
                    "factor_variacion": variacion,
                },
                convergencia_objetivo="derivacion_coherente",
            )
            trayectorias.append(trayectoria_derivado)

        trayectorias.insert(0, trayectoria_padre)
        return trayectorias

    def procesar_bifurcaciones_activas(self, G: nx.Graph, paso_actual: int) -> Dict[str, Any]:
        """Procesa todas las bifurcaciones activas en el paso actual"""
        resultados = {
            "trayectorias_procesadas": 0,
            "convergencias_detectadas": 0,
            "bifurcaciones_completadas": [],
            "nuevos_nodos_generados": [],
        }

        bifurcaciones_completadas: List[Any] = []

        for nodo_id, espacio_bifurcacion in list(self.bifurcaciones_activas.items()):
            try:
                pasos_transcurridos = paso_actual - espacio_bifurcacion.paso_inicio

                if pasos_transcurridos < espacio_bifurcacion.pasos_exploracion:
                    resultado_evolucion = self._evolucionar_trayectorias(espacio_bifurcacion, pasos_transcurridos, G)
                    resultados["trayectorias_procesadas"] += resultado_evolucion["procesadas"]
                else:
                    resultado_convergencia = self._converger_bifurcacion(espacio_bifurcacion, G, nodo_id)

                    if resultado_convergencia["exitosa"]:
                        resultados["convergencias_detectadas"] += 1
                        resultados["bifurcaciones_completadas"].append(nodo_id)
                        resultados["nuevos_nodos_generados"].extend(resultado_convergencia.get("nodos_generados", []))
                        bifurcaciones_completadas.append(nodo_id)
                        self.estadisticas_bifurcacion["convergencias_exitosas"] += 1

            except Exception:
                bifurcaciones_completadas.append(nodo_id)

        for nodo_id in bifurcaciones_completadas:
            self.bifurcaciones_activas.pop(nodo_id, None)

        return resultados

    def _evolucionar_trayectorias(self, espacio_bifurcacion: EspacioBifurcacion, paso_relativo: int, G: nx.Graph) -> Dict[str, int]:
        """Evoluciona las trayectorias de una bifurcación en el paso actual"""
        resultado = {"procesadas": 0, "colapsadas": 0}

        for trayectoria in espacio_bifurcacion.trayectorias:
            if not trayectoria.activa:
                continue
            try:
                if paso_relativo < len(trayectoria.secuencia_glifica):
                    glifo_actual = trayectoria.secuencia_glifica[paso_relativo]
                    self._aplicar_transformacion_trayectoria(trayectoria, glifo_actual, G, espacio_bifurcacion.nodo_origen_id)
                    trayectoria.pasos_completados += 1
                    resultado["procesadas"] += 1

                viabilidad = self._evaluar_viabilidad_trayectoria(trayectoria)
                trayectoria.viabilidad = viabilidad

                if viabilidad < 0.2:
                    trayectoria.activa = False
                    resultado["colapsadas"] += 1
                    self.estadisticas_bifurcacion["trayectorias_colapsadas"] += 1
            except Exception:
                trayectoria.activa = False
                resultado["colapsadas"] += 1

        return resultado

    def _aplicar_transformacion_trayectoria(self, trayectoria: TrayectoriaBifurcacion, glifo: str, G: nx.Graph, nodo_origen_id: Any) -> None:
        """Aplica una transformación glífica específica a una trayectoria"""
        try:
            if nodo_origen_id not in G.nodes:
                return

            if glifo == "VAL":
                factor = 1.15 if trayectoria.tipo == "expansion_coherente" else 1.05
                trayectoria.parametros_iniciales["EPI"] = _safe_float(trayectoria.parametros_iniciales.get("EPI", 1.0), 1.0) * factor
            elif glifo == "NUL":
                factor = 0.85 if trayectoria.tipo == "contraccion_resonante" else 0.95
                trayectoria.parametros_iniciales["EPI"] = _safe_float(trayectoria.parametros_iniciales.get("EPI", 1.0), 1.0) * factor
            elif glifo == "ZHIR":
                trayectoria.parametros_iniciales["EPI"] = _safe_float(trayectoria.parametros_iniciales.get("EPI", 1.0), 1.0) + 0.5
                trayectoria.parametros_iniciales["νf"] = _safe_float(trayectoria.parametros_iniciales.get("νf", 1.0), 1.0) * 1.2
            elif glifo == "RA":
                trayectoria.parametros_iniciales["Si"] = _safe_float(trayectoria.parametros_iniciales.get("Si", 0.5), 0.5) * 1.1
            elif glifo == "IL":
                epi_objetivo = 1.5
                epi_actual = _safe_float(trayectoria.parametros_iniciales.get("EPI", 1.0), 1.0)
                trayectoria.parametros_iniciales["EPI"] = epi_actual * 0.8 + epi_objetivo * 0.2
            elif glifo == "THOL":
                for param in ["EPI", "νf", "Si"]:
                    if param in trayectoria.parametros_iniciales:
                        v = _safe_float(trayectoria.parametros_iniciales[param], 1.0)
                        trayectoria.parametros_iniciales[param] = max(0.8, min(2.0, v))
        except Exception:
            pass

    def _evaluar_viabilidad_trayectoria(self, trayectoria: TrayectoriaBifurcacion) -> float:
        """Evalúa la viabilidad estructural de una trayectoria"""
        try:
            epi = _safe_float(trayectoria.parametros_iniciales.get("EPI", 1.0), 1.0)
            vf = _safe_float(trayectoria.parametros_iniciales.get("νf", 1.0), 1.0)
            si = _safe_float(trayectoria.parametros_iniciales.get("Si", 0.5), 0.5)

            if not all(_safe_isfinite(x) for x in [epi, vf, si]):
                return 0.0

            criterios = []
            criterios.append(1.0 if 0.3 <= epi <= 3.5 else 0.0)  # rango estructural
            criterios.append(1.0 if 0.2 <= vf <= 2.5 else 0.0)   # frecuencia resonante
            criterios.append(1.0 if si >= 0.1 else 0.0)          # coherencia mínima
            ratio_equilibrio = min(epi / max(vf, 0.001), vf / max(epi, 0.001))
            criterios.append(ratio_equilibrio)
            progreso = trayectoria.pasos_completados / max(len(trayectoria.secuencia_glifica), 1)
            criterios.append(min(progreso, 1.0))

            viabilidad = float(np.mean(criterios))
            return max(0.0, min(1.0, viabilidad))
        except Exception:
            return 0.0

    def _converger_bifurcacion(self, espacio_bifurcacion: EspacioBifurcacion, G: nx.Graph, nodo_origen_id: Any) -> Dict[str, Any]:
        """Convierte el espacio de bifurcación en configuración final estable"""
        resultado = {"exitosa": False, "nodos_generados": [], "configuracion_final": None}
        try:
            trayectorias_viables = [t for t in espacio_bifurcacion.trayectorias if t.activa and t.viabilidad > 0.3]
            if not trayectorias_viables:
                return resultado

            trayectorias_viables.sort(key=lambda t: t.viabilidad, reverse=True)
            if len(trayectorias_viables) == 1:
                resultado = self._aplicar_trayectoria_ganadora(trayectorias_viables[0], G, nodo_origen_id)
            else:
                resultado = self._fusionar_trayectorias_compatibles(trayectorias_viables[:3], G, nodo_origen_id)

            if resultado["exitosa"]:
                self.convergencias_detectadas.append({
                    "nodo_origen": nodo_origen_id,
                    "tipo_bifurcacion": espacio_bifurcacion.tipo_bifurcacion,
                    "trayectorias_fusionadas": len(trayectorias_viables),
                    "configuracion_final": resultado["configuracion_final"],
                })
            return resultado
        except Exception:
            return resultado

    def _aplicar_trayectoria_ganadora(self, trayectoria: TrayectoriaBifurcacion, G: nx.Graph, nodo_origen_id: Any) -> Dict[str, Any]:
        """Aplica la configuración de una trayectoria ganadora al nodo origen"""
        try:
            if nodo_origen_id not in G.nodes:
                return {"exitosa": False}

            nodo_data = G.nodes[nodo_origen_id]
            for param, valor in trayectoria.parametros_iniciales.items():
                if param in ["EPI", "νf", "Si", "ΔNFR"] and _safe_isfinite(valor):
                    nodo_data[param] = max(0.1, min(3.0, _safe_float(valor, 1.0)))

            nodo_data["ultima_bifurcacion"] = {
                "tipo": trayectoria.tipo,
                "convergencia": trayectoria.convergencia_objetivo,
                "viabilidad_final": trayectoria.viabilidad,
            }
            return {"exitosa": True, "configuracion_final": dict(trayectoria.parametros_iniciales), "nodos_generados": [nodo_origen_id]}
        except Exception:
            return {"exitosa": False}

    def _fusionar_trayectorias_compatibles(self, trayectorias: List[TrayectoriaBifurcacion], G: nx.Graph, nodo_origen_id: Any) -> Dict[str, Any]:
        """Fusiona múltiples trayectorias compatibles en una configuración híbrida"""
        try:
            if nodo_origen_id not in G.nodes:
                return {"exitosa": False}

            total_viabilidad = sum(t.viabilidad for t in trayectorias)
            if total_viabilidad == 0:
                return {"exitosa": False}

            configuracion_fusionada: Dict[str, float] = {}
            for param in ["EPI", "νf", "Si", "ΔNFR"]:
                suma_ponderada = sum(_safe_float(t.parametros_iniciales.get(param, 1.0), 1.0) * t.viabilidad for t in trayectorias)
                valor_fusionado = suma_ponderada / total_viabilidad
                if _safe_isfinite(valor_fusionado):
                    configuracion_fusionada[param] = max(0.1, min(3.0, valor_fusionado))

            nodo_data = G.nodes[nodo_origen_id]
            for param, valor in configuracion_fusionada.items():
                nodo_data[param] = valor

            nodo_data["ultima_bifurcacion"] = {
                "tipo": "fusion_multiple",
                "trayectorias_fusionadas": len(trayectorias),
                "viabilidad_promedio": total_viabilidad / len(trayectorias),
            }
            return {"exitosa": True, "configuracion_final": configuracion_fusionada, "nodos_generados": [nodo_origen_id]}
        except Exception:
            return {"exitosa": False}

    def obtener_estadisticas_bifurcacion(self) -> Dict[str, Any]:
        """Retorna estadísticas completas del sistema de bifurcaciones"""
        stats = self.estadisticas_bifurcacion.copy()
        stats.update({
            "bifurcaciones_activas": len(self.bifurcaciones_activas),
            "tasa_convergencia": (stats["convergencias_exitosas"] / max(stats["total_bifurcaciones"], 1)),
            "tasa_colapso": (stats["trayectorias_colapsadas"] / max(stats["total_bifurcaciones"] * 2.5, 1)),
        })
        return stats


def integrar_bifurcaciones_canonicas_en_simulacion(G: nx.Graph, paso: int, coordinador_temporal: TemporalCoordinatorTNFR, bifurcation_manager: BifurcationManagerTNFR) -> Dict[str, Any]:
    """Integra el sistema de bifurcaciones canónicas en OntoSim."""
    resultados = {
        "nuevas_bifurcaciones": 0,
        "trayectorias_procesadas": 0,
        "convergencias_completadas": 0,
        "nodos_modificados": [],
    }
    try:
        if hasattr(bifurcation_manager, "bifurcaciones_activas"):
            res_proc = bifurcation_manager.procesar_bifurcaciones_activas(G, paso)
            resultados["trayectorias_procesadas"] = res_proc["trayectorias_procesadas"]
            resultados["convergencias_completadas"] = res_proc["convergencias_detectadas"]
            resultados["nodos_modificados"].extend(res_proc["nuevos_nodos_generados"])

        for nodo_id, nodo_data in G.nodes(data=True):
            if nodo_data.get("estado") != "activo":
                continue
            if nodo_id in bifurcation_manager.bifurcaciones_activas:
                continue

            es_bif, tipo = bifurcation_manager.detectar_bifurcacion_canonica(nodo_data, nodo_id)
            if es_bif and tipo != "error_deteccion":
                espacio = bifurcation_manager.generar_espacio_bifurcacion(nodo_id, nodo_data, tipo, paso)
                if espacio:
                    bifurcation_manager.bifurcaciones_activas[nodo_id] = espacio
                    resultados["nuevas_bifurcaciones"] += 1
                    resultados["nodos_modificados"].append(nodo_id)
        return resultados
    except Exception:
        return resultados


def reemplazar_deteccion_bifurcacion_simple(G: nx.Graph, paso: int, umbrales: Dict[str, float], bifurcation_manager: BifurcationManagerTNFR) -> List[tuple]:
    """Reemplaza la detección simple de bifurcaciones en OntoSim."""
    nodos_bifurcados: List[tuple] = []
    try:
        umbral_aceleracion = _safe_float(umbrales.get("bifurcacion_umbral", 0.15), 0.15)
        for nodo_id, nodo_data in G.nodes(data=True):
            if nodo_data.get("estado") != "activo":
                continue
            if nodo_id in bifurcation_manager.bifurcaciones_activas:
                continue
            es_bif, tipo = bifurcation_manager.detectar_bifurcacion_canonica(nodo_data, nodo_id, umbral_aceleracion)
            if es_bif:
                nodos_bifurcados.append((nodo_id, tipo))
        return nodos_bifurcados
    except Exception:
        return []


def mostrar_trayectorias_activas(bifurcation_manager: BifurcationManagerTNFR) -> str:
    """Muestra detalles de las trayectorias activas"""
    if not bifurcation_manager.bifurcaciones_activas:
        return "No hay bifurcaciones activas"
    detalles = []
    for nodo_id, espacio in bifurcation_manager.bifurcaciones_activas.items():
        trayectorias_activas = [t for t in espacio.trayectorias if t.activa]
        viabilidades = [f"{t.viabilidad:.2f}" for t in trayectorias_activas]
        detalles.append(f"  {nodo_id}: {espacio.tipo_bifurcacion} (" f"{len(trayectorias_activas)} activas, viabilidades: {viabilidades})")
    return "Trayectorias activas:\n" + "\n".join(detalles)


def limpiar_bifurcaciones_obsoletas(bifurcation_manager: BifurcationManagerTNFR, paso_actual: int, limite_pasos: int = 50) -> int:
    """Limpia bifurcaciones que han excedido el tiempo máximo de exploración"""
    obsoletas: List[Any] = []
    for nodo_id, espacio in list(bifurcation_manager.bifurcaciones_activas.items()):
        if (paso_actual - espacio.paso_inicio) > limite_pasos:
            obsoletas.append(nodo_id)
    for nodo_id in obsoletas:
        bifurcation_manager.bifurcaciones_activas.pop(nodo_id, None)
    return len(obsoletas)


# =========================================================================================
# SISTEMA DE UMBRALES DINÁMICOS (corregido)
# =========================================================================================

def calcular_umbrales_dinamicos(C_t: float, densidad_nodal: float, fase_simulacion: str = "emergencia") -> Dict[str, float]:
    # Sensibilidad por desviación de C(t) del equilibrio
    equilibrio_base = 1.0
    C_t_safe = _safe_float(C_t, equilibrio_base)
    densidad_safe = _safe_float(densidad_nodal, 3.0)
    
    desviacion_C_t = abs(C_t_safe - equilibrio_base)

    sensibilidad = max(0.4, min(2.0, 1.0 + 0.8 * desviacion_C_t))
    factor_densidad = max(0.7, min(1.5, 1.0 - 0.1 * (densidad_safe - 3.0)))

    multiplicadores_fase = {
        "emergencia": 1.2,
        "estabilizacion": 0.8,
        "bifurcacion": 1.5,
    }
    factor_fase = multiplicadores_fase.get(fase_simulacion, 1.0)

    sensibilidad_final = sensibilidad * factor_densidad * factor_fase

    return {
        # Umbrales de conexión (para crear/eliminar aristas)
        "θ_conexion": 0.12 * sensibilidad_final,
        "EPI_conexion": 1.8 * sensibilidad_final,
        "νf_conexion": 0.2 * sensibilidad_final,
        "Si_conexion": 0.25 * sensibilidad_final,
        # Umbrales críticos nodales (en fase θ)
        "θ_mutacion": 0.25 * sensibilidad_final,        # activa ZHIR si |Δθ| supera este valor
        "θ_colapso": 0.45 * sensibilidad_final,         # reserva para colapsos por fase (no se usa como gradiente)
        "θ_autoorganizacion": 0.35 * sensibilidad_final,
        # Límites de estabilidad estructural
        "EPI_max_dinamico": max(2.5, C_t_safe * 2.8),
        "EPI_min_coherencia": max(0.4, C_t_safe * 0.3),
        # Umbrales de bifurcación estructural
        "bifurcacion_aceleracion": 0.15 * sensibilidad_final,
        "bifurcacion_gradiente": 0.8 * sensibilidad_final,
        # Metadatos
        "C_t_usado": C_t_safe,
        "sensibilidad_calculada": sensibilidad_final,
        "factor_densidad": factor_densidad,
        "fase": fase_simulacion,
    }


def evaluar_condiciones_emergencia_dinamica(nodo: Dict[str, Any], umbrales: Dict[str, float], campo_coherencia: float) -> tuple[bool, str]:
    """Evalúa si un nodo cumple condiciones de emergencia con umbrales dinámicos."""
    epi = _safe_float(nodo.get("EPI", 0), 0)
    vf = _safe_float(nodo.get("νf", 0), 0)
    theta_nodo = _safe_float(nodo.get("θ", 0.5), 0.5)
    dNFR_abs = abs(_safe_float(nodo.get("ΔNFR", 0), 0))
    campo_coherencia_safe = _safe_float(campo_coherencia, 1.0)
    
    if epi < _safe_float(umbrales.get("EPI_min_coherencia", 0.4), 0.4):
        return False, f"EPI insuficiente: {epi:.3f} < {umbrales.get('EPI_min_coherencia', 0.4):.3f}"

    if vf < 0.3:
        return False, f"Frecuencia demasiado baja: {vf:.3f}"

    # **Usar θ (fase) canónica**
    if abs(theta_nodo - 0.0) > 0.7 and campo_coherencia_safe > 1.2:
        return False, f"Disonancia con campo: θ={theta_nodo:.3f}, C(t)={campo_coherencia_safe:.3f}"

    if dNFR_abs > _safe_float(umbrales.get("bifurcacion_gradiente", 0.8), 0.8):
        return False, f"Gradiente excesivo: {dNFR_abs:.3f} > {umbrales.get('bifurcacion_gradiente', 0.8):.3f}"

    return True, "Condiciones de emergencia cumplidas"


def detectar_fase_simulacion(G: nx.Graph, paso_actual: int, historial_C_t: List[tuple[int, float]], ventana: int = 50) -> str:
    """Detecta la fase actual de la simulación: "emergencia", "estabilizacion" o "bifurcacion"."""
    if len(historial_C_t) < ventana:
        return "emergencia"

    valores_recientes = [_safe_float(c_t, 1.0) for _, c_t in historial_C_t[-ventana:]]
    variabilidad = float(np.std(valores_recientes)) if valores_recientes else 0.0
    tendencia = float(np.mean(valores_recientes[-10:]) - np.mean(valores_recientes[:10])) if len(valores_recientes) >= 20 else 0.0

    nodos_activos = sum(1 for n in G.nodes if G.nodes[n].get("estado") == "activo")
    fraccion_activa = nodos_activos / len(G.nodes) if len(G.nodes) else 0.0

    if variabilidad > 0.3 and abs(tendencia) > 0.2:
        return "bifurcacion"
    elif variabilidad < 0.05 and fraccion_activa > 0.6:
        return "estabilizacion"
    else:
        return "emergencia"


def aplicar_umbrales_dinamicos_conexiones(G: nx.Graph, umbrales: Dict[str, float]) -> Dict[str, Any]:
    """Aplica umbrales dinámicos para gestión de conexiones de red (con barra de progreso)."""
    conexiones_creadas = 0
    conexiones_eliminadas = 0
    nodos_lista = list(G.nodes)
    total_nodos = len(nodos_lista)

    total_pares = (total_nodos * (total_nodos - 1)) // 2
    pbar = tqdm(total=total_pares, desc="Evaluando conexiones TNFR", unit="pares", dynamic_ncols=True)

    for i in range(total_nodos):
        for j in range(i + 1, total_nodos):
            n1, n2 = nodos_lista[i], nodos_lista[j]
            nodo1, nodo2 = G.nodes[n1], G.nodes[n2]

            condiciones_resonancia = [
                abs(_safe_float(nodo1.get("θ", 0), 0) - _safe_float(nodo2.get("θ", 0), 0)) < _safe_float(umbrales.get("θ_conexion", 0.12), 0.12),
                abs(_safe_float(nodo1.get("EPI", 0), 0) - _safe_float(nodo2.get("EPI", 0), 0)) < _safe_float(umbrales.get("EPI_conexion", 1.8), 1.8),
                abs(_safe_float(nodo1.get("νf", 1), 1) - _safe_float(nodo2.get("νf", 1), 1)) < _safe_float(umbrales.get("νf_conexion", 0.2), 0.2),
                abs(_safe_float(nodo1.get("Si", 0), 0) - _safe_float(nodo2.get("Si", 0), 0)) < _safe_float(umbrales.get("Si_conexion", 0.25), 0.25),
            ]
            resonancia_suficiente = sum(condiciones_resonancia) >= 3

            vecinos_n1 = len(list(G.neighbors(n1)))
            vecinos_n2 = len(list(G.neighbors(n2)))
            max_conexiones = int(8 * _safe_float(umbrales.get("sensibilidad_calculada", 1.0), 1.0))

            saturacion = vecinos_n1 >= max_conexiones and vecinos_n2 >= max_conexiones
            existe_conexion = G.has_edge(n1, n2)

            if resonancia_suficiente and not saturacion and not existe_conexion:
                G.add_edge(n1, n2)
                conexiones_creadas += 1
            elif not resonancia_suficiente and existe_conexion:
                G.remove_edge(n1, n2)
                conexiones_eliminadas += 1

            pbar.update(1)

    pbar.close()

    return {
        "conexiones_creadas": conexiones_creadas,
        "conexiones_eliminadas": conexiones_eliminadas,
        "umbrales_usados": umbrales,
    }


def evaluar_activacion_glifica_dinamica(nodo: Dict[str, Any], umbrales: Dict[str, float], vecinos_data: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """Evalúa qué glifo debería activarse basado en umbrales dinámicos."""
    # Z'HIR — Mutación por salto de fase
    theta_actual = _safe_float(nodo.get("θ", 0.0), 0.0)
    theta_prev = _safe_float(nodo.get("θ_prev", theta_actual), theta_actual)
    if abs(theta_actual - theta_prev) > _safe_float(umbrales.get("θ_mutacion", 0.25), 0.25):
        return "ZHIR"

    # SH'A — Colapso por baja coherencia + gradiente alto (usa umbral de GRADIENTE)
    epi = _safe_float(nodo.get("EPI", 0.0), 0.0)
    dnfr = abs(_safe_float(nodo.get("ΔNFR", 0.0), 0.0))
    if (epi < _safe_float(umbrales.get("EPI_min_coherencia", 0.4), 0.4)
            and dnfr > _safe_float(umbrales.get("bifurcacion_gradiente", 0.8), 0.8)):
        return "SHA"

    # T'HOL — Autoorganización por aceleración estructural
    aceleracion = abs(_safe_float(nodo.get("d2EPI_dt2", 0.0), 0.0))
    if aceleracion > _safe_float(umbrales.get("bifurcacion_aceleracion", 0.15), 0.15):
        return "THOL"

    # R'A — Resonancia con vecinos (si están muy sincronizados en θ)
    if vecinos_data:
        thetas = [_safe_float(v.get("θ", 0.0), 0.0) for v in vecinos_data]
        if thetas:
            resonancia_promedio = sum(abs(theta_actual - tv) for tv in thetas) / len(thetas)
            if resonancia_promedio < _safe_float(umbrales.get("θ_conexion", 0.12), 0.12) * 0.5:
                return "RA"

    return None


def gestionar_conexiones_canonico(G: nx.Graph, paso: int, historia_Ct: List[tuple[int, float]]):
    """Gestión de conexiones canónica TNFR (para reemplazar la lógica manual en OntoSim)."""
    C_t = sum(_safe_float(G.nodes[n].get("EPI", 1.0), 1.0) for n in G.nodes) / len(G) if len(G.nodes) else 0.0
    densidad_promedio = (sum(len(list(G.neighbors(n))) for n in G.nodes) / len(G.nodes)) if len(G.nodes) else 0.0
    fase_actual = detectar_fase_simulacion(G, paso, historia_Ct)
    umbrales = calcular_umbrales_dinamicos(C_t, densidad_promedio, fase_actual)
    estadisticas = aplicar_umbrales_dinamicos_conexiones(G, umbrales)
    return umbrales, estadisticas


# =========================================================================================
# EXPORTS
# =========================================================================================

__all__ = [
    # Funciones seguras agregadas
    "_safe_isfinite",
    "_safe_float",
    
    # Ecuación nodal + clamps + medición
    "actualizar_EPI_por_ecuacion_nodal",
    "aplicar_clamps_canonicos",
    "medir_C",

    # Temporal
    "TemporalCoordinatorTNFR",
    "inicializar_coordinador_temporal_canonico",
    "integrar_tiempo_topologico_en_simulacion",

    # Bifurcaciones
    "EspacioBifurcacion",
    "BifurcationManagerTNFR",
    "integrar_bifurcaciones_canonicas_en_simulacion",
    "reemplazar_deteccion_bifurcacion_simple",
    "mostrar_trayectorias_activas",
    "limpiar_bifurcaciones_obsoletas",

    # Umbrales/activación/conexiones
    "calcular_umbrales_dinamicos",
    "aplicar_umbrales_dinamicos_conexiones",
    "evaluar_condiciones_emergencia_dinamica",
    "evaluar_activacion_glifica_dinamica",
    "gestionar_conexiones_canonico",
    "detectar_fase_simulacion",
]