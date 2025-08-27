import networkx as nx
from collections import deque

from tnfr.constants import attach_defaults
from tnfr.operators import aplicar_remesh_si_estabilizacion_global


def test_aplicar_remesh_usa_parametro_personalizado():
    G = nx.Graph()
    G.add_node(0)
    attach_defaults(G)

    # Historial suficiente para el parámetro personalizado
    hist = G.graph.setdefault("history", {})
    hist["stable_frac"] = [1.0, 1.0, 1.0]

    # Historial de EPI necesario para aplicar_remesh_red
    tau = G.graph["REMESH_TAU"]
    maxlen = max(2 * tau + 5, 64)
    G.graph["_epi_hist"] = deque([{0: 0.0} for _ in range(tau + 1)], maxlen=maxlen)

    # Sin parámetro personalizado no se debería activar
    aplicar_remesh_si_estabilizacion_global(G)
    assert "_last_remesh_step" not in G.graph

    # Con parámetro personalizado se activa con 3 pasos estables
    aplicar_remesh_si_estabilizacion_global(G, pasos_estables_consecutivos=3)
    assert G.graph["_last_remesh_step"] == len(hist["stable_frac"])

