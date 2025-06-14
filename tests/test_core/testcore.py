import pytest
import networkx as nx
from tnfr.core.ontosim import (
    inicializar_nfr_emergente,
    crear_red_desde_datos,
    simular_emergencia,
    cumple_condiciones_emergencia,
    evaluar_coherencia_estructural
)

def test_inicializar_nfr_valores():
    nfr = inicializar_nfr_emergente('coherencia_test')
    assert nfr is not None
    assert nfr['estado'] == 'activo'
    assert 'EPI' in nfr
    assert 0.5 <= nfr['EPI'] <= 2.5

def test_crear_red_con_datos():
    datos = [
        {'id': 'nodo1', 'forma_base': 'coherencia'},
        {'id': 'nodo2', 'forma_base': 'resonancia'}
    ]
    G = crear_red_desde_datos(datos)
    assert len(G.nodes) == 2
    assert 'nodo1' in G.nodes
    assert G.nodes['nodo1']['estado'] == 'activo'

def test_simular_emergencia_basica():
    datos = [
        {'id': 'nodo1', 'forma_base': 'coherencia'},
        {'id': 'nodo2', 'forma_base': 'resonancia'}
    ]
    G = crear_red_desde_datos(datos)
    historia, red_final, epis, lecturas, _, _, _, _ = simular_emergencia(G, pasos=5)
    assert len(historia) == 5
    assert len(red_final.nodes) >= 2
    assert isinstance(epis, list)

def test_cumple_condiciones():
    assert cumple_condiciones_emergencia('coherencia', None) == True
    assert cumple_condiciones_emergencia('', None) == False

def test_evaluar_coherencia():
    resultado = evaluar_coherencia_estructural('coherencia')
    assert isinstance(resultado, float)
