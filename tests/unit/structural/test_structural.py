"""Pruebas de structural."""

import networkx as nx
import pytest

from tnfr import run_sequence
from tnfr.structural import (
    create_nfr,
    Operator,
    Emission,
    Reception,
    Coherence,
    Dissonance,
    Coupling,
    Resonance,
    Silence,
    Expansion,
    Contraction,
    SelfOrganization,
    Mutation,
    Transition,
    Recursivity,
    validate_sequence,
)
from tnfr.operators.compat import (
    Emision,
    Recepcion,
    Coherencia,
    Resonancia,
    Silencio,
    Autoorganizacion,
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
    EMISSION,
    RECEPTION,
    COHERENCE,
    RESONANCE,
    SILENCE,
    SELF_ORGANIZATION,
    TRANSITION,
)


def test_create_nfr_basic():
    G, n = create_nfr("nodo", epi=0.1, vf=2.0, theta=0.3)
    assert isinstance(G, nx.Graph)
    assert n in G
    nd = G.nodes[n]
    assert nd[EPI_PRIMARY] == 0.1


def test_sequence_validation_and_run():
    G, n = create_nfr("x")
    ops = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    names = [op.name for op in ops]
    ok, msg = validate_sequence(names)
    assert ok, msg
    run_sequence(G, n, ops)
    # después de la secuencia la EPI se actualiza (no necesariamente cero)
    assert EPI_PRIMARY in G.nodes[n]


def test_legacy_sequence_validation_and_run_warns():
    G, n = create_nfr("x_es")
    with pytest.warns(DeprecationWarning):
        ops = [Emision(), Recepcion(), Coherencia(), Resonancia(), Silencio()]
    names = [op.name for op in ops]
    ok, msg = validate_sequence(names)
    assert ok, msg
    run_sequence(G, n, ops)
    assert EPI_PRIMARY in G.nodes[n]


def test_invalid_sequence():
    with pytest.warns(DeprecationWarning):
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


def test_thol_requires_closure_english_tokens():
    names = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        SELF_ORGANIZATION,
        RESONANCE,
        TRANSITION,
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
    with pytest.warns(DeprecationWarning):
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


def test_thol_closed_by_silence_alias():
    ops = [
        Emission(),
        Reception(),
        Coherence(),
        SelfOrganization(),
        Resonance(),
        Silence(),
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


def test_sequence_accepts_english_tokens():
    names = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        RESONANCE,
        SILENCE,
    ]
    ok, msg = validate_sequence(names)
    assert ok, msg


def test_operator_base_types_exposed():
    for cls in (
        Emission,
        Reception,
        Coherence,
        Dissonance,
        Coupling,
        Resonance,
        Silence,
        Expansion,
        Contraction,
        SelfOrganization,
        Mutation,
        Transition,
        Recursivity,
    ):
        assert issubclass(cls, Operator)
    with pytest.warns(DeprecationWarning):
        legacy = Autoorganizacion()
    assert isinstance(legacy, Operator)
