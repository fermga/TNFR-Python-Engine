"""Pruebas de remesh."""
from collections import deque

from tnfr.constants import attach_defaults
from tnfr.operators import apply_remesh_if_globally_stable


def test_aplicar_remesh_usa_parametro_personalizado(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    G.graph["REMESH_REQUIRE_STABILITY"] = False

    # Historial suficiente para el parámetro personalizado
    hist = G.graph.setdefault("history", {})
    hist["stable_frac"] = [1.0, 1.0, 1.0]

    # Historial de EPI necesario para apply_network_remesh
    tau = G.graph["REMESH_TAU_GLOBAL"]
    maxlen = max(2 * tau + 5, 64)
    G.graph["_epi_hist"] = deque([{0: 0.0} for _ in range(tau + 1)], maxlen=maxlen)

    # Sin parámetro personalizado no se debería activar
    apply_remesh_if_globally_stable(G)
    assert "_last_remesh_step" not in G.graph

    # Con parámetro personalizado se activa con 3 pasos estables
    apply_remesh_if_globally_stable(G, pasos_estables_consecutivos=3)
    assert G.graph["_last_remesh_step"] == len(hist["stable_frac"])


def test_remesh_alpha_hard_ignores_glyph_factor(graph_canon):
    G = graph_canon()
    G.add_node(0)
    attach_defaults(G)
    G.graph["REMESH_REQUIRE_STABILITY"] = False
    hist = G.graph.setdefault("history", {})
    hist["stable_frac"] = [1.0, 1.0, 1.0]
    tau = G.graph["REMESH_TAU_GLOBAL"]
    maxlen = max(2 * tau + 5, 64)
    G.graph["_epi_hist"] = deque([{0: 0.0} for _ in range(tau + 1)], maxlen=maxlen)
    G.graph["REMESH_ALPHA"] = 0.7
    G.graph["REMESH_ALPHA_HARD"] = True
    G.graph["GLYPH_FACTORS"]["REMESH_alpha"] = 0.1
    apply_remesh_if_globally_stable(G, pasos_estables_consecutivos=3)
    meta = G.graph.get("_REMESH_META", {})
    assert meta.get("alpha") == 0.7
    assert G.graph.get("_REMESH_ALPHA_SRC") == "REMESH_ALPHA"

