"""Pruebas de structural."""

import networkx as nx

from tnfr import run_sequence
from tnfr.structural import (
    create_nfr,
    Emision,
    Recepcion,
    Coherencia,
    Resonancia,
    Silencio,
    Autoorganizacion,
    validate_sequence,
)
from tnfr.constants import EPI_PRIMARY
from tnfr.config.operator_names import (
    EMISION,
    RECEPCION,
    COHERENCIA,
    RESONANCIA,
    SILENCIO,
    AUTOORGANIZACION,
    TRANSICION,
)


def test_create_nfr_basic():
    G, n = create_nfr("nodo", epi=0.1, vf=2.0, theta=0.3)
    assert isinstance(G, nx.Graph)
    assert n in G
    nd = G.nodes[n]
    assert nd[EPI_PRIMARY] == 0.1


def test_sequence_validation_and_run():
    G, n = create_nfr("x")
    ops = [Emision(), Recepcion(), Coherencia(), Resonancia(), Silencio()]
    names = [op.name for op in ops]
    ok, msg = validate_sequence(names)
    assert ok, msg
    run_sequence(G, n, ops)
    # después de la secuencia la EPI se actualiza (no necesariamente cero)
    assert EPI_PRIMARY in G.nodes[n]


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


def test_thol_requires_closure():
    names = [
        EMISION,
        RECEPCION,
        COHERENCIA,
        AUTOORGANIZACION,
        RESONANCIA,
        TRANSICION,
    ]
    ok, msg = validate_sequence(names)
    assert not ok


def test_validate_sequence_rejects_unknown_tokens():
    names = [
        EMISION,
        RECEPCION,
        COHERENCIA,
        RESONANCIA,
        SILENCIO,
        "desconocido",
    ]
    ok, msg = validate_sequence(names)
    assert not ok and "unknown tokens" in msg


def test_thol_closed_by_silencio():
    ops = [
        Emision(),
        Recepcion(),
        Coherencia(),
        Autoorganizacion(),
        Resonancia(),
        Silencio(),
    ]
    names = [op.name for op in ops]
    ok, msg = validate_sequence(names)
    assert ok, msg


def test_sequence_rejects_trailing_tokens():
    names = [
        EMISION,
        RECEPCION,
        COHERENCIA,
        RESONANCIA,
        SILENCIO,
        EMISION,
    ]
    ok, msg = validate_sequence(names)
    assert not ok
