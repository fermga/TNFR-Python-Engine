import sys, pathlib

import networkx as nx

# Ensure module path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from tnfr.operators import aplicar_remesh_si_estabilizacion_global


def test_remesh_uses_custom_stable_steps():
    G = nx.path_graph(3)
    # Configure defaults with a window larger than our custom value
    G.graph['REMESH_STABILITY_WINDOW'] = 5
    G.graph['FRACTION_STABLE_REMESH'] = 0.8
    # Simulated history reaching stability
    G.graph['history'] = {'stable_frac': [0.9, 0.9, 0.9]}

    aplicar_remesh_si_estabilizacion_global(G, pasos_estables_consecutivos=3)

    assert G.graph['_last_remesh_step'] == 3
