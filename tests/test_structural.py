"""Pruebas de structural."""
import networkx as nx

from tnfr.structural import (
    create_nfr,
    Emision,
    Recepcion,
    Coherencia,
    Resonancia,
    Silencio,
    validate_sequence,
    run_sequence,
)
from tnfr.constants import ALIAS_EPI


def test_create_nfr_basic():
    G, n = create_nfr("nodo", epi=0.1, vf=2.0, theta=0.3)
    assert isinstance(G, nx.Graph)
    assert n in G
    nd = G.nodes[n]
    assert nd[ALIAS_EPI[0]] == 0.1


def test_sequence_validation_and_run():
    G, n = create_nfr("x")
    ops = [Emision(), Recepcion(), Coherencia(), Resonancia(), Silencio()]
    names = [op.name for op in ops]
    ok, msg = validate_sequence(names)
    assert ok, msg
    run_sequence(G, n, ops)
    # después de la secuencia la EPI se actualiza (no necesariamente cero)
    assert ALIAS_EPI[0] in G.nodes[n]


def test_invalid_sequence():
    ops = [Recepcion(), Coherencia(), Silencio()]
    names = [op.name for op in ops]
    ok, msg = validate_sequence(names)
    assert not ok
    G, n = create_nfr("y")
    try:
        run_sequence(G, n, ops)
    except ValueError:
        pass
    else:
        raise AssertionError("Se esperaba ValueError por secuencia no válida")

