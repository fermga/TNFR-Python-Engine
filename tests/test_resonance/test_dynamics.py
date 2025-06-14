import pytest
from tnfr.resonance.dynamics import (
    TemporalCoordinatorTNFR,
    BifurcationManagerTNFR,
    calcular_umbrales_dinamicos
)

class TestTemporalCoordinator:
    def test_inicializacion(self):
        coord = TemporalCoordinatorTNFR()
        assert coord.sincronizacion_global == True
        assert coord.frecuencia_pulsos == 50

    def test_calcular_paso_temporal(self):
        coord = TemporalCoordinatorTNFR()
        nodo = {'νf': 1.0, 'Si': 0.5, 'θ': 0.3, 'estado': 'activo'}
        paso = coord.calcular_paso_temporal_nodal(nodo, 10)
        assert isinstance(paso, float)

class TestBifurcationManager:
    def test_inicializacion(self):
        gestor = BifurcationManagerTNFR()
        assert isinstance(gestor.bifurcaciones_activas, dict)

    def test_detectar_bifurcacion(self):
        gestor = BifurcationManagerTNFR()
        nodo = {'d2EPI_dt2': 0.2, 'DeltaNFR': 0.9, 'Si': 0.6, 'EPI': 1.5, 'νf': 1.0}
        es_bif, tipo = gestor.detectar_bifurcacion_canonica(nodo, 'nodo_test')
        assert isinstance(es_bif, bool)

def test_calcular_umbrales():
    umbrales = calcular_umbrales_dinamicos(C_t=1.0, densidad_nodal=3.0, fase_simulacion="emergencia")

    # Verifica todas las claves esperadas
    claves_esperadas = [
        'θ_conexion',
        'EPI_conexion',
        'νf_conexion',
        'Si_conexion',
        'θ_mutacion',
        'θ_colapso',
        'θ_autoorganizacion',
        'EPI_max_dinamico',
        'EPI_min_coherencia',
        'bifurcacion_aceleracion',
        'bifurcacion_gradiente',
        'C_t_usado',
        'sensibilidad_calculada',
        'factor_densidad',
        'fase'
    ]
    for clave in claves_esperadas:
        assert clave in umbrales

    # Verifica que la fase sea la correcta
    assert umbrales['fase'] == 'emergencia'

    # Verifica valores calculados (opcional, para robustez)
    assert umbrales['θ_conexion'] == pytest.approx(0.12 * 1.2, rel=1e-6)
    assert umbrales['EPI_conexion'] == pytest.approx(1.8 * 1.2, rel=1e-6)
    assert umbrales['C_t_usado'] == 1.0

    # Prueba con otra fase
    umbrales_estabilizacion = calcular_umbrales_dinamicos(C_t=1.0, densidad_nodal=3.0, fase_simulacion="estabilizacion")
    assert umbrales_estabilizacion['fase'] == 'estabilizacion'
    assert umbrales_estabilizacion['θ_conexion'] < umbrales['θ_conexion']  # Debe ser menor por el factor fase
