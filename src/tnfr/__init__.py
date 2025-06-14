"""
TNFR - Teoría de la Naturaleza Fractal Resonante
Simulador simbólico de coherencia emergente
"""

__version__ = "0.1.0"

# Importar TODO usando * con __all__ controlado
from .core.ontosim import *
from .matrix.operators import *
from .resonance.dynamics import *
from .utils.helpers import *

# Funciones de conveniencia
def crear_simulacion_basica(datos_nodos, pasos=250):
    """Función de conveniencia para crear y ejecutar una simulación TNFR básica."""
    red = crear_red_desde_datos(datos_nodos)
    historia, red_final, epis, lecturas, _, historial_glifos, _, stats = simular_emergencia(red, pasos)
    
    return {
        'historia': historia,
        'red_final': red_final,
        'epis_compuestas': epis,
        'lecturas_sintacticas': lecturas,
        'historial_glifos': historial_glifos,
        'estadisticas_bifurcacion': stats
    }

def version_info():
    """Retorna información del paquete TNFR."""
    return {
        'version': __version__,
        'modulos': ['core', 'matrix', 'resonance', 'utils']
    }
