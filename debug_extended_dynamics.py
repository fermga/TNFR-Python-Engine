"""Debug de la dinámica extendida para identificar por qué no evoluciona."""

import sys
import os
import networkx as nx

# Agregar el src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tnfr.dynamics.integrators import _update_extended_nodal_system
from tnfr.structural import create_nfr
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR


def debug_extended_dynamics():
    """Debug paso a paso de la dinámica extendida."""
    print("🔍 DEBUG: Dinámica extendida paso a paso")
    print("=" * 60)
    
    # Crear red conectada
    G = nx.connected_watts_strogatz_graph(4, 2, 0.3, seed=123)
    
    # Inicializar nodos con valores que generen flujos
    for i, node in enumerate(G.nodes()):
        G.nodes[node][ALIAS_EPI] = 0.3 + i * 0.2  # Variación para gradientes
        G.nodes[node][ALIAS_VF] = 0.8 + i * 0.4    # Diferentes frecuencias
        G.nodes[node][ALIAS_DNFR] = 0.1 * (-1)**i  # ΔNFR alternante
        G.nodes[node]['theta'] = i * 0.5           # Fases diferentes
    
    print(f"Red: {G.number_of_nodes()} nodos, {G.number_of_edges()} enlaces")
    print()
    
    # Estado inicial
    print("ESTADO INICIAL:")
    for node in G.nodes():
        nd = G.nodes[node]
        print(f"  Nodo {node}:")
        print(f"    EPI: {nd[ALIAS_EPI]:.3f}")
        print(f"    νf: {nd[ALIAS_VF]:.3f}")
        print(f"    ΔNFR: {nd[ALIAS_DNFR]:.3f}")
        print(f"    θ: {nd['theta']:.3f}")
    
    print()
    
    # Test disponibilidad de campos extendidos
    print("VERIFICACIÓN CAMPOS EXTENDIDOS:")
    try:
        from tnfr.physics.extended_canonical_fields import (
            compute_phase_current,
            compute_dnfr_flux
        )
        print("  ✅ Módulos de campos disponibles")
        
        # Test con primer nodo
        node_test = list(G.nodes())[0]
        
        try:
            j_phi_result = compute_phase_current(G, node_test)
            print(f"  ✅ J_φ computado: {j_phi_result}")
        except Exception as e:
            print(f"  ❌ Error en J_φ: {e}")
            
        try:
            j_dnfr_result = compute_dnfr_flux(G, node_test)
            print(f"  ✅ J_ΔNFR computado: {j_dnfr_result}")
        except Exception as e:
            print(f"  ❌ Error en J_ΔNFR: {e}")
            
    except ImportError as e:
        print(f"  ❌ Módulos no disponibles: {e}")
    
    print()
    
    # Test sistema canónico extendido
    print("VERIFICACIÓN SISTEMA CANÓNICO:")
    try:
        from tnfr.dynamics.canonical import compute_extended_nodal_system
        
        # Parámetros de test
        result = compute_extended_nodal_system(
            nu_f=1.0,
            delta_nfr=0.1, 
            theta=0.5,
            j_phi=0.05,  # Flujo no-cero
            j_dnfr_divergence=-0.02,  # Divergencia no-cero
            coupling_strength=0.8
        )
        
        print("  ✅ Sistema canónico funciona:")
        print(f"    ∂EPI/∂t: {result.classical_derivative:.6f}")
        print(f"    ∂θ/∂t: {result.phase_derivative:.6f}")
        print(f"    ∂ΔNFR/∂t: {result.dnfr_derivative:.6f}")
        
    except Exception as e:
        print(f"  ❌ Error en sistema canónico: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test integración completa  
    print("INTEGRACIÓN COMPLETA:")
    try:
        G.graph['use_extended_dynamics'] = True
        dt = 0.05
        
        # Estado antes
        node_0 = list(G.nodes())[0]
        epi_before = G.nodes[node_0][ALIAS_EPI]
        theta_before = G.nodes[node_0]['theta']
        dnfr_before = G.nodes[node_0][ALIAS_DNFR]
        
        print(f"  Antes - EPI: {epi_before:.6f}, θ: {theta_before:.6f}, ΔNFR: {dnfr_before:.6f}")
        
        # Integrar
        _update_extended_nodal_system(G, dt=dt)
        
        # Estado después
        epi_after = G.nodes[node_0][ALIAS_EPI]
        theta_after = G.nodes[node_0]['theta']
        dnfr_after = G.nodes[node_0][ALIAS_DNFR]
        
        print(f"  Después - EPI: {epi_after:.6f}, θ: {theta_after:.6f}, ΔNFR: {dnfr_after:.6f}")
        
        # Cambios
        delta_epi = epi_after - epi_before
        delta_theta = theta_after - theta_before
        delta_dnfr = dnfr_after - dnfr_before
        
        print(f"  Cambios - ΔEPI: {delta_epi:.6f}, Δθ: {delta_theta:.6f}, ΔΔNFR: {delta_dnfr:.6f}")
        
        # Verificar derivadas cacheadas
        if 'dtheta_dt' in G.nodes[node_0]:
            print(f"  Derivadas - ∂EPI/∂t: {G.nodes[node_0].get('dEPI', 'N/A'):.6f}")
            print(f"              ∂θ/∂t: {G.nodes[node_0]['dtheta_dt']:.6f}")  
            print(f"              ∂ΔNFR/∂t: {G.nodes[node_0]['ddnfr_dt']:.6f}")
        
        # Diagnóstico
        total_change = abs(delta_epi) + abs(delta_theta) + abs(delta_dnfr)
        if total_change > 1e-6:
            print("  ✅ Sistema evolucionando correctamente")
        else:
            print("  ⚠️ Sistema en equilibrio (cambios muy pequeños)")
            
            # Posibles causas
            print("  Posibles causas de equilibrio:")
            print("    - Flujos J_φ, J_ΔNFR muy pequeños")
            print("    - Coeficientes de física muy conservadores")
            print("    - Red en configuración estable")
            
    except Exception as e:
        print(f"  ❌ Error en integración: {e}")
        import traceback
        traceback.print_exc()


def test_manual_extended_system():
    """Test manual con flujos forzados grandes."""
    print("\n" + "=" * 60)
    print("🧪 TEST MANUAL: Flujos forzados grandes")
    print("=" * 60)
    
    try:
        from tnfr.dynamics.canonical import compute_extended_nodal_system
        
        # Test con flujos grandes para ver evolución clara
        result = compute_extended_nodal_system(
            nu_f=2.0,           # Frecuencia alta
            delta_nfr=0.3,      # ΔNFR grande
            theta=1.0,          # Fase intermedia
            j_phi=0.5,          # Flujo de fase grande
            j_dnfr_divergence=-0.2,  # Divergencia grande
            coupling_strength=1.5    # Acoplamiento fuerte
        )
        
        print(f"Parámetros grandes:")
        print(f"  νf: 2.0, ΔNFR: 0.3, θ: 1.0")
        print(f"  J_φ: 0.5, ∇·J_ΔNFR: -0.2, κ: 1.5")
        print()
        print(f"Resultados:")
        print(f"  ∂EPI/∂t = {result.classical_derivative:.6f}")
        print(f"  ∂θ/∂t = {result.phase_derivative:.6f}")
        print(f"  ∂ΔNFR/∂t = {result.dnfr_derivative:.6f}")
        
        # Con dt = 0.1, cambios esperados:
        dt = 0.1
        expected_delta_epi = result.classical_derivative * dt
        expected_delta_theta = result.phase_derivative * dt
        expected_delta_dnfr = result.dnfr_derivative * dt
        
        print()
        print(f"Con dt = {dt}, cambios esperados:")
        print(f"  ΔEPI ≈ {expected_delta_epi:.6f}")
        print(f"  Δθ ≈ {expected_delta_theta:.6f}")
        print(f"  ΔΔNFR ≈ {expected_delta_dnfr:.6f}")
        
        # Verificar que hay evolución significativa
        total_expected = abs(expected_delta_epi) + abs(expected_delta_theta) + abs(expected_delta_dnfr)
        
        if total_expected > 1e-3:
            print(f"  ✅ Evolución significativa esperada (total: {total_expected:.6f})")
        else:
            print(f"  ⚠️ Evolución muy pequeña (total: {total_expected:.6f})")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_extended_dynamics()
    test_manual_extended_system()