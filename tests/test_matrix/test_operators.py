import pytest
import networkx as nx
from tnfr.constants import glifo_categoria
from tnfr.matrix.operators import (
    aplicar_glifo,
    evaluar_patron_glifico,
    normalizar_historial_glifos
)

def test_operadores_definidos():
    """Verifica que todos los operadores estén definidos."""
    operadores_esperados = [
        "AL", "EN", "IL", "OZ", "UM", "RA", "SHA", 
        "VAL", "NUL", "THOL", "ZHIR", "NAV", "REMESH"
    ]
    for op in operadores_esperados:
        assert op in glifo_categoria
    assert len(glifo_categoria) == 13

def test_aplicar_glifo_al():
    G = nx.Graph()
    G.add_node('test', EPI=1.0, glifo=None, Si=0.5, νf=1.0, ΔNFR=0.2)
    historial = {}
    nodo = G.nodes['test']  # <-- Aquí defines nodo
    if 'Si' not in nodo:
        nodo['Si'] = 0.5
    aplicar_glifo(G, nodo, 'test', "AL", historial, 1)
    assert G.nodes['test']['glifo'] == 'AL'
    assert 'test' in historial

def test_evaluar_patron_glifico_simple():
    """Prueba evaluación de patrón simple."""
    patron = evaluar_patron_glifico(["AL", "UM", "RA"])
    assert patron['inicio_creativo'] == True
    assert "AL → UM → RA" in patron['patron_glifico']

def test_normalizar_historial():
    historial = {'nodo1': [(1, 'AL'), (2, 'UM')]}
    normalizado = normalizar_historial_glifos(historial)
    assert normalizado == [(1, 'AL'), (2, 'UM')]