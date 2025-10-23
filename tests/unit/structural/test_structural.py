"""Structural tests covering canonical operator wiring."""

import networkx as nx
import pytest

from tnfr import run_sequence
from tnfr.config.operator_names import (
    COHERENCE,
    EMISSION,
    RECEPTION,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from tnfr.constants import EPI_PRIMARY
from tnfr.structural import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Operator,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
    create_nfr,
    validate_sequence,
)


def test_create_nfr_basic() -> None:
    G, n = create_nfr("node", epi=0.1, vf=2.0, theta=0.3)
    assert isinstance(G, nx.Graph)
    assert n in G
    nd = G.nodes[n]
    assert nd[EPI_PRIMARY] == 0.1


def test_sequence_validation_and_run() -> None:
    G, n = create_nfr("x")
    ops = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    names = [op.name for op in ops]
    ok, msg = validate_sequence(names)
    assert ok, msg
    run_sequence(G, n, ops)
    assert EPI_PRIMARY in G.nodes[n]


def test_invalid_sequence() -> None:
    ops = [Reception(), Coherence(), Silence()]
    names = [op.name for op in ops]
    ok, msg = validate_sequence(names)
    assert not ok
    G, n = create_nfr("y")
    with pytest.raises(ValueError):
        run_sequence(G, n, ops)


def test_thol_requires_closure() -> None:
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


def test_validate_sequence_rejects_unknown_tokens() -> None:
    names = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        RESONANCE,
        SILENCE,
        "unknown",
    ]
    ok, msg = validate_sequence(names)
    assert not ok and "unknown tokens" in msg


def test_thol_closed_by_silence() -> None:
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


def test_sequence_rejects_trailing_tokens() -> None:
    names = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        RESONANCE,
        SILENCE,
        EMISSION,
    ]
    ok, msg = validate_sequence(names)
    assert not ok


def test_sequence_accepts_english_tokens() -> None:
    names = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        RESONANCE,
        SILENCE,
    ]
    ok, msg = validate_sequence(names)
    assert ok, msg


def test_validate_sequence_rejects_legacy_keyword() -> None:
    with pytest.raises(TypeError) as excinfo:
        validate_sequence(legacy_names=[EMISSION, RECEPTION, COHERENCE])

    assert "unexpected keyword argument" in str(excinfo.value)


def test_operator_base_types_exposed() -> None:
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
