"""Memoized preflight must preserve canonical grammar and live metadata."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tnfr.operators.definitions import (
    Coherence,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Reception,
    Resonance,
    SelfOrganization,
    Silence,
)
from tnfr.operators.grammar_core import GrammarValidator
from tnfr.operators.grammar_memoization import (
    clear_memoization_cache,
    create_sequence_signature,
    get_memoization_stats,
    validate_sequence_optimized,
)
from tnfr.validation.compatibility import CompatibilityLevel


def test_default_arguments_validate_a_canonical_word() -> None:
    sequence = [Emission(), Coherence(), Silence()]
    assert validate_sequence_optimized(sequence)[0]


@pytest.mark.parametrize("compatibility", list(CompatibilityLevel))
@pytest.mark.parametrize(
    "sequence",
    [
        [Emission(), Mutation(), Coherence(), Silence()],
        [Emission(), Coherence(), SelfOrganization(), Silence()],
        [Emission(), Coherence(), Dissonance(), Mutation(), Coherence(), Silence()],
        [Emission(), Coherence(), Dissonance(), SelfOrganization(), Silence()],
    ],
)
def test_transformer_context_matches_canonical_validator(
    compatibility: CompatibilityLevel, sequence: list
) -> None:
    expected, _ = GrammarValidator().validate(sequence)
    actual, _ = validate_sequence_optimized(sequence, compatibility_level=compatibility)
    assert actual is expected


@pytest.mark.parametrize("epi_initial", [0.0, -0.1, 1e-12, 1.0])
def test_initiation_boundary_matches_canonical_validator(epi_initial: float) -> None:
    sequence = [Coherence(), Silence()]
    expected, _ = GrammarValidator().validate(sequence, epi_initial=epi_initial)
    actual, _ = validate_sequence_optimized(
        sequence, epi_initial=epi_initial, compatibility_level=CompatibilityLevel.GOOD
    )
    assert actual is expected


def test_cache_does_not_freeze_multiscale_metadata() -> None:
    clear_memoization_cache()
    recursive = SimpleNamespace(name="recursivity", depth=1)
    sequence = [Emission(), recursive, Silence()]
    assert validate_sequence_optimized(sequence)[0]
    recursive.depth = 3
    valid, messages = validate_sequence_optimized(sequence)
    assert not valid
    assert any("U5 violated" in message for message in messages)
    assert get_memoization_stats()["static_validation_cache"]["hits"] == 1


def test_external_context_does_not_override_ordered_u4b() -> None:
    sequence = [Emission(), Mutation(), Coherence(), Silence()]
    valid, _ = validate_sequence_optimized(
        sequence,
        recent_destabilizers=["OZ"],
        bifurcation_window=100,
        compatibility_level=CompatibilityLevel.GOOD,
    )
    assert not valid


def test_prior_coherence_need_not_be_inside_destabilizer_window() -> None:
    """U4b requires recent pressure, while the prior IL supplies a stable base."""
    sequence = [
        Emission(),
        Coherence(),
        Reception(),
        Reception(),
        Expansion(),
        Mutation(),
        Coherence(),
        Silence(),
    ]
    assert GrammarValidator().validate(sequence)[0]
    assert validate_sequence_optimized(sequence)[0]


def test_canonical_names_and_glyphs_share_signature() -> None:
    assert create_sequence_signature([Emission(), Coherence(), Silence()]) == (
        create_sequence_signature(["AL", "IL", "SHA"])
    )
    assert validate_sequence_optimized(["emission", "coherence", "silence"])[0]


def test_coupling_readout_still_requires_runtime_phase_check() -> None:
    import networkx as nx

    valid, messages = validate_sequence_optimized(
        [Emission(), Resonance(), Silence()],
        graph=nx.Graph(),
        compatibility_level=CompatibilityLevel.GOOD,
    )
    assert valid
    assert any(
        "Phase compatibility validation required" in message for message in messages
    )


def test_unknown_operator_does_not_pass_static_preflight() -> None:
    valid, messages = validate_sequence_optimized(["NOT_AN_OPERATOR", "SHA"])
    assert not valid
    assert any("Unknown operator" in message for message in messages)
