"""Prototipo experimental: Sistema de ecuaciones TNFR extendido con flujos canónicos.

Este módulo implementa un prototipo de la extensión fundamental de TNFR 
que incluye J_φ (corriente de fase) y J_ΔNFR (flujo de reorganización) 
como campos canónicos que afectan la evolución del sistema.

ADVERTENCIA: Este es código experimental. No usar en producción.

Referencias:
- analysis_integracion_sistemica_campos_canonicos.md
- notebooks/Extended_Fields_Investigation.ipynb (células 35-40)
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, Any
from dataclasses import dataclass

# Importaciones TNFR existentes
from tnfr.physics.fields import compute_phase_gradient
from tnfr.physics.extended_canonical_fields import compute_phase_current, compute_dnfr_flux
from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR


@dataclass
class ExtendedNodalResult:
    """Resultado del sistema nodal extendido."""
    depi_dt: float      # ∂EPI/∂t (clásico)
    dtheta_dt: float    # ∂θ/∂t (nuevo: evolución fase con transporte) 
    ddnfr_dt: float     # ∂ΔNFR/∂t (nuevo: conservación de reorganización)
    j_phi: float        # Corriente de fase J_φ
    j_dnfr_div: float   # Divergencia flujo ΔNFR: ∇·J_ΔNFR


def compute_phase_transport_evolution(
    nu_f: float, 
    delta_nfr: float, 
    j_phi: float,
    coupling_strength: float = 0.1
) -> float:
    """Compute ∂θ/∂t con transporte de fase.
    
    Nueva ecuación extendida:
    ∂θ/∂t = α·νf·sin(θ) + β·ΔNFR + γ·J_φ
    
    Donde:
    - Término clásico: α·νf·sin(θ) (autoorganización)
    - Término de presión: β·ΔNFR (respuesta a reorganización)  
    - Término de transporte: γ·J_φ (flujo dirigido de fase)
    """
    # Coeficientes experimentales (ajustar según validación)
    alpha = 0.5   # Acoplamiento νf-θ
    beta = 0.3    # Sensibilidad a ΔNFR
    gamma = 0.2   # Eficiencia de transporte J_φ
    
    # Término clásico: autoorganización de fase
    classical_term = alpha * nu_f * np.sin(np.pi * delta_nfr)  # sin(θ) aproximado
    
    # Término de presión: respuesta a reorganización
    pressure_term = beta * delta_nfr
    
    # Término de transporte: flujo dirigido de fase
    transport_term = gamma * j_phi * coupling_strength
    
    return classical_term + pressure_term + transport_term


def compute_dnfr_conservation_evolution(
    j_dnfr_divergence: float,
    decay_rate: float = 0.05
) -> float:
    """Compute ∂ΔNFR/∂t por conservación de flujo.
    
    Nueva ecuación de conservación:
    ∂ΔNFR/∂t = -∇·J_ΔNFR - λ·ΔNFR
    
    Donde:
    - Término de conservación: -∇·J_ΔNFR (flujo neto)
    - Término de decaimiento: -λ·ΔNFR (relajación natural)
    """
    # Conservación: flujo entrante aumenta ΔNFR, saliente lo disminuye
    conservation_term = -j_dnfr_divergence
    
    # Decaimiento natural hacia equilibrio
    decay_term = -decay_rate * np.abs(j_dnfr_divergence) * np.sign(j_dnfr_divergence)
    
    return conservation_term + decay_term


def compute_extended_nodal_system(
    G: nx.Graph,
    node: Any,
    dt: float = 0.01
) -> ExtendedNodalResult:
    """Compute sistema nodal extendido con flujos canónicos.
    
    Sistema de ecuaciones acopladas:
    1. ∂EPI/∂t = νf · ΔNFR(t)           [Ecuación nodal clásica]
    2. ∂θ/∂t = f(νf, ΔNFR, J_φ)        [Evolución fase con transporte]  
    3. ∂ΔNFR/∂t = g(∇·J_ΔNFR)          [Conservación reorganización]
    """
    
    # Obtener estado actual del nodo
    epi = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
    nu_f = get_attr(G.nodes[node], ALIAS_VF, 1.0)
    delta_nfr = get_attr(G.nodes[node], ALIAS_DNFR, 0.0)
    theta = G.nodes[node].get('theta', 0.0)
    
    # Compute campos canónicos extendidos
    try:
        j_phi_result = compute_phase_current(G, node)
        j_dnfr_result = compute_dnfr_flux(G, node)
        
        # Extraer valores escalares
        if isinstance(j_phi_result, dict):
            j_phi = j_phi_result.get('J_phi', 0.0)
        else:
            j_phi = float(j_phi_result)
            
        if isinstance(j_dnfr_result, dict):
            j_dnfr = j_dnfr_result.get('J_dnfr', 0.0)
        else:
            j_dnfr = float(j_dnfr_result)
        
        # Aproximar divergencia como diferencia con vecinos
        j_dnfr_div = approximate_flux_divergence(G, node, 'j_dnfr')
        
    except Exception as e:
        # Fallback si campos no están disponibles
        print(f"Warning: Using fallback for node {node}: {e}")
        j_phi = 0.0
        j_dnfr = 0.0
        j_dnfr_div = 0.0
    
    # 1. Ecuación nodal clásica (sin cambios)
    depi_dt = nu_f * delta_nfr
    
    # 2. Nueva evolución de fase con transporte
    coupling_strength = estimate_local_coupling_strength(G, node)
    dtheta_dt = compute_phase_transport_evolution(nu_f, delta_nfr, j_phi, coupling_strength)
    
    # 3. Nueva conservación de ΔNFR
    ddnfr_dt = compute_dnfr_conservation_evolution(j_dnfr_div)
    
    return ExtendedNodalResult(
        depi_dt=depi_dt,
        dtheta_dt=dtheta_dt, 
        ddnfr_dt=ddnfr_dt,
        j_phi=j_phi,
        j_dnfr_div=j_dnfr_div
    )


def approximate_flux_divergence(G: nx.Graph, node: Any, flux_attr: str) -> float:
    """Aproximar ∇·J usando diferencias finitas con vecinos."""
    if G.degree(node) == 0:
        return 0.0
    
    # Flujo del nodo central
    central_flux = G.nodes[node].get(flux_attr, 0.0)
    
    # Flujo promedio de vecinos
    neighbor_fluxes = [
        G.nodes[neighbor].get(flux_attr, 0.0) 
        for neighbor in G.neighbors(node)
    ]
    
    if not neighbor_fluxes:
        return 0.0
        
    mean_neighbor_flux = np.mean(neighbor_fluxes)
    
    # Divergencia aproximada: diferencia normalizada
    divergence = (central_flux - mean_neighbor_flux) / np.sqrt(G.degree(node))
    
    return divergence


def estimate_local_coupling_strength(G: nx.Graph, node: Any) -> float:
    """Estimar fuerza de acoplamiento local basada en topología."""
    degree = G.degree(node)
    if degree == 0:
        return 0.0
    
    # Acoplamiento aumenta con conectividad, pero satura
    max_degree = 10.0  # Saturación
    normalized_degree = min(degree / max_degree, 1.0)
    
    # Curva sigmoidal para acoplamiento realista
    coupling = 1.0 / (1.0 + np.exp(-5 * (normalized_degree - 0.5)))
    
    return coupling


def update_extended_system(
    G: nx.Graph,
    dt: float = 0.01,
    validation_mode: bool = True
) -> Dict[str, Any]:
    """Actualizar todo el sistema con dinámica extendida.
    
    Esta función reemplazaría a update_epi_via_nodal_equation()
    en un sistema TNFR completamente extendido.
    """
    
    results = {}
    conservation_violations = []
    
    for node in G.nodes():
        
        # Compute sistema extendido para este nodo
        result = compute_extended_nodal_system(G, node, dt)
        
        # Aplicar updates
        current_epi = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
        current_theta = G.nodes[node].get('theta', 0.0)
        current_dnfr = get_attr(G.nodes[node], ALIAS_DNFR, 0.0)
        
        # Integrate ecuaciones
        new_epi = current_epi + result.depi_dt * dt
        new_theta = (current_theta + result.dtheta_dt * dt) % (2 * np.pi)
        new_dnfr = current_dnfr + result.ddnfr_dt * dt
        
        # Aplicar clipping para estabilidad
        new_epi = np.clip(new_epi, 0.0, 1.0)
        new_dnfr = np.clip(new_dnfr, -2.0, 2.0)  # Límites conservadores
        
        # Update nodo
        set_attr(G.nodes[node], ALIAS_EPI, new_epi)
        G.nodes[node]['theta'] = new_theta
        set_attr(G.nodes[node], ALIAS_DNFR, new_dnfr)
        
        # Store resultados para análisis
        results[node] = {
            'depi_dt': result.depi_dt,
            'dtheta_dt': result.dtheta_dt, 
            'ddnfr_dt': result.ddnfr_dt,
            'j_phi': result.j_phi,
            'j_dnfr_div': result.j_dnfr_div,
            'new_epi': new_epi,
            'new_theta': new_theta,
            'new_dnfr': new_dnfr
        }
        
        # Validación de conservación
        if validation_mode and abs(result.j_dnfr_div) > 0.5:  # Umbral
            conservation_violations.append({
                'node': node,
                'j_dnfr_div': result.j_dnfr_div,
                'ddnfr_dt': result.ddnfr_dt
            })
    
    # Summary stats
    all_j_phi = [r['j_phi'] for r in results.values()]
    all_j_dnfr_div = [r['j_dnfr_div'] for r in results.values()]
    
    summary = {
        'nodes_updated': len(results),
        'mean_j_phi': np.mean(all_j_phi),
        'std_j_phi': np.std(all_j_phi),
        'mean_j_dnfr_div': np.mean(all_j_dnfr_div),
        'std_j_dnfr_div': np.std(all_j_dnfr_div),
        'conservation_violations': len(conservation_violations),
        'violation_details': conservation_violations[:5]  # Primeros 5
    }
    
    return {
        'node_results': results,
        'summary': summary,
        'dt_used': dt
    }


def validate_classical_limit(G: nx.Graph, tolerance: float = 1e-6) -> bool:
    """Validar que J→0 recupera la dinámica clásica TNFR.
    
    Test crítico: cuando J_φ = J_ΔNFR = 0, el sistema debe
    reducirse exactamente a ∂EPI/∂t = νf·ΔNFR.
    """
    
    # Forzar flujos a cero
    for node in G.nodes():
        G.nodes[node]['j_phi'] = 0.0
        G.nodes[node]['j_dnfr'] = 0.0
    
    # Compute con sistema extendido
    extended_results = {}
    for node in G.nodes():
        result = compute_extended_nodal_system(G, node)
        extended_results[node] = result
    
    # Compute con sistema clásico
    classical_results = {}
    for node in G.nodes():
        nu_f = get_attr(G.nodes[node], ALIAS_VF, 1.0)
        delta_nfr = get_attr(G.nodes[node], ALIAS_DNFR, 0.0)
        classical_depi_dt = nu_f * delta_nfr
        classical_results[node] = classical_depi_dt
    
    # Comparar ∂EPI/∂t
    violations = []
    for node in G.nodes():
        extended_depi = extended_results[node].depi_dt
        classical_depi = classical_results[node]
        error = abs(extended_depi - classical_depi)
        
        if error > tolerance:
            violations.append({
                'node': node,
                'extended': extended_depi,
                'classical': classical_depi,
                'error': error
            })
    
    # Validar que dtheta_dt y ddnfr_dt son pequeños cuando J=0
    phase_violations = []
    dnfr_violations = []
    
    for node in G.nodes():
        result = extended_results[node]
        
        if abs(result.dtheta_dt) > tolerance:
            phase_violations.append({
                'node': node, 
                'dtheta_dt': result.dtheta_dt
            })
            
        if abs(result.ddnfr_dt) > tolerance:
            dnfr_violations.append({
                'node': node,
                'ddnfr_dt': result.ddnfr_dt  
            })
    
    is_valid = (len(violations) == 0 and 
                len(phase_violations) == 0 and 
                len(dnfr_violations) == 0)
    
    if not is_valid:
        print(f"Classical limit validation FAILED:")
        print(f"  EPI violations: {len(violations)}")
        print(f"  Phase violations: {len(phase_violations)}")  
        print(f"  DNFR violations: {len(dnfr_violations)}")
        if violations:
            print(f"  Sample EPI error: {violations[0]}")
    
    return is_valid


# Funciones de utilidad para testing
def create_test_network_extended(n_nodes: int = 10, seed: int = 42) -> nx.Graph:
    """Crear red de test con campos extendidos inicializados."""
    np.random.seed(seed)
    
    # Crear grafo conectado
    G = nx.connected_watts_strogatz_graph(n_nodes, 4, 0.3, seed=seed)
    
    # Inicializar atributos TNFR clásicos
    for node in G.nodes():
        G.nodes[node][ALIAS_EPI] = np.random.uniform(0.2, 0.8)
        G.nodes[node][ALIAS_VF] = np.random.uniform(0.5, 2.0) 
        G.nodes[node][ALIAS_DNFR] = np.random.uniform(-0.1, 0.1)
        G.nodes[node]['theta'] = np.random.uniform(0, 2*np.pi)
        
        # Inicializar campos extendidos (pequeños para validación)
        G.nodes[node]['j_phi'] = np.random.uniform(-0.01, 0.01)
        G.nodes[node]['j_dnfr'] = np.random.uniform(-0.01, 0.01)
    
    return G


if __name__ == "__main__":
    print("=== PROTOTIPO: Sistema TNFR Extendido ===")
    print()
    
    # Test 1: Límite clásico
    print("Test 1: Validación límite clásico")
    G = create_test_network_extended(n_nodes=5)
    
    is_classical_valid = validate_classical_limit(G)
    print(f"  ✅ Límite clásico: {'PASS' if is_classical_valid else 'FAIL'}")
    print()
    
    # Test 2: Evolución extendida
    print("Test 2: Evolución con flujos extendidos")
    G = create_test_network_extended(n_nodes=5)
    
    # Estado inicial
    initial_state = {
        node: {
            'epi': get_attr(G.nodes[node], ALIAS_EPI, 0.0),
            'theta': G.nodes[node].get('theta', 0.0),
            'dnfr': get_attr(G.nodes[node], ALIAS_DNFR, 0.0)
        }
        for node in G.nodes()
    }
    
    # Evolución
    result = update_extended_system(G, dt=0.05)
    
    print(f"  Nodos actualizados: {result['summary']['nodes_updated']}")
    print(f"  J_φ promedio: {result['summary']['mean_j_phi']:.6f}")
    print(f"  ∇·J_ΔNFR promedio: {result['summary']['mean_j_dnfr_div']:.6f}")
    print(f"  Violaciones conservación: {result['summary']['conservation_violations']}")
    
    # Verificar cambios
    changes = []
    for node in G.nodes():
        old_epi = initial_state[node]['epi']
        new_epi = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
        change = abs(new_epi - old_epi)
        changes.append(change)
    
    print(f"  Cambio EPI promedio: {np.mean(changes):.6f}")
    print(f"  ✅ Evolución: {'PASS' if np.mean(changes) > 1e-6 else 'FAIL'}")
    
    print()
    print("🎯 Prototipo completado. Ver analysis_integracion_sistemica_campos_canonicos.md")
    print("   para próximos pasos de integración sistemática.")