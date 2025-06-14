import pytest
import networkx as nx
from tnfr.core.ontosim import crear_red_desde_datos, inicializar_nfr_emergente

@pytest.fixture
def datos_nodos_test():
    """Datos de prueba para crear nodos TNFR"""
    return [
        {'id': 'test_nodo1', 'forma_base': 'coherencia'},
        {'id': 'test_nodo2', 'forma_base': 'resonancia'},
        {'id': 'test_nodo3', 'forma_base': 'mutacion'}
    ]

@pytest.fixture
def red_tnfr_basica(datos_nodos_test):
    """Red TNFR básica para pruebas"""
    return crear_red_desde_datos(datos_nodos_test)
