import pytest
import networkx as nx
from tnfr.utils.helpers import (
    emergencia_nodal,
    promover_emergente,
    detectar_nodos_pulsantes,
    detectar_macronodos,
    obtener_nodos_emitidos,
    evaluar_si_nodal,
    exportar_nodos_emitidos
)

def test_emergencia_nodal():
    nodo = {'EPI': 1.0, 'Si': 0.5, 'νf': 1.0, 'ΔNFR': 0.05}
    resultado = emergencia_nodal(nodo, 1.0, 0.1)
    assert isinstance(resultado, bool)

def test_promover_emergente():
    G = nx.Graph()
    G.add_node('nodo1', EPI=1.0, estado='activo', Si=0.5, νf=1.0, θ=0.3, ΔNFR=0.0)
    historial = {}
    promover_emergente('nodo1', G, 1, historial, [])
    assert 'nodo1' in historial

def test_detectar_nodos_pulsantes():
    # Ejemplo con 3 ciclos "VAL" → "NUL"
    historial = {
        'nodo1': [
            (1, 'VAL'), (2, 'NUL'),  # ciclo 1
            (3, 'VAL'), (4, 'NUL'),  # ciclo 2
            (5, 'VAL'), (6, 'NUL')   # ciclo 3
        ]
    }
    pulsantes = detectar_nodos_pulsantes(historial)
    assert isinstance(pulsantes, list)
    assert 'nodo1' in pulsantes

    # Ejemplo sin suficientes ciclos
    historial_malo = {
        'nodo2': [
            (1, 'VAL'), (2, 'NUL'),  # ciclo 1
            (3, 'VAL'), (4, 'UM')    # no es ciclo
        ]
    }
    pulsantes = detectar_nodos_pulsantes(historial_malo)
    assert 'nodo2' not in pulsantes

def test_obtener_nodos_emitidos():
    G = nx.Graph()
    G.add_node(
        'nodo1',
        emitido=True,
        glifo='AL',
        categoria='alguna_categoria',
        EPI=1.0,
        Si=0.5,
        ΔNFR=0.0,
        θ=0.3,
        νf=1.0
    )
    emitidos, detalles = obtener_nodos_emitidos(G)
    assert 'nodo1' in emitidos

def test_evaluar_si_nodal():
    """Prueba evaluación de coherencia nodal."""
    nodo = {'Si': 0.5}
    resultado = evaluar_si_nodal(nodo)
    assert isinstance(resultado, float)

def test_exportar_nodos_emitidos():
    G = nx.Graph()
    G.add_node(
        'nodo1',
        emitido=True,
        glifo='AL',
        categoria='alguna_categoria',
        EPI=1.0,
        Si=0.5,
        ΔNFR=0.0,
        θ=0.3,
        νf=1.0
    )
    resultado = exportar_nodos_emitidos(G, archivo='nodos_emitidos.json')
    assert isinstance(resultado, dict)
    assert resultado['exitosa'] is True
    assert resultado['nodos_exportados'] == 1
    assert resultado['archivo'] == 'nodos_emitidos.json'
