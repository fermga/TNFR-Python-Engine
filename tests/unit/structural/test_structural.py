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
    operator_display_name,
)
from tnfr.constants import (
    D2EPI_PRIMARY,
    DNFR_PRIMARY,
    EPI_KIND_PRIMARY,
    EPI_PRIMARY,
    THETA_PRIMARY,
    VF_PRIMARY,
)
from tnfr.mathematics import BasicStateProjector
from tnfr.validation import NFRValidator, SequenceValidationResult
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
    create_math_nfr,
    create_nfr,
    validate_sequence,
)
from tnfr.validation import ValidationOutcome

def _outcome_stub(names: list[str] | tuple[str, ...]) -> SequenceValidationResult:
    sequence = tuple(names)
    return SequenceValidationResult(
        tokens=sequence,
        canonical_tokens=sequence,
        passed=True,
        message="ok",
        metadata={},
    )

def test_create_nfr_basic() -> None:
    G, n = create_nfr("node", epi=0.1, vf=2.0, theta=0.3)
    assert isinstance(G, nx.Graph)
    assert n in G
    nd = G.nodes[n]
    assert nd[EPI_PRIMARY] == 0.1

def test_create_nfr_installs_default_dnfr_hook() -> None:
    G, _ = create_nfr("seed-default")
    hook = G.graph.get("compute_delta_nfr")

    assert callable(hook)
    assert G.graph["_dnfr_hook_name"] == "dnfr_epi_vf_mixed"

def test_create_nfr_accepts_custom_dnfr_hook() -> None:
    calls: list[tuple[nx.Graph, int | None]] = []

    def stub(graph: nx.Graph, *, n_jobs: int | None = None) -> None:  # pragma: no cover - signature only
        calls.append((graph, n_jobs))

    G, _ = create_nfr("seed-custom", dnfr_hook=stub)
    hook = G.graph["compute_delta_nfr"]
    hook(G, n_jobs=5)

    assert calls == [(G, 5)]
    assert hook.__name__ == stub.__name__
    assert G.graph["_dnfr_hook_name"] == stub.__name__

def test_sequence_validation_and_run() -> None:
    G, n = create_nfr("x")
    ops = [Emission(), Reception(), Coherence(), Resonance(), Silence()]
    names = [op.name for op in ops]
    outcome = validate_sequence(names)
    assert isinstance(outcome, ValidationOutcome)
    assert isinstance(outcome, SequenceValidationResult)
    assert outcome.passed, outcome.summary["message"]
    assert outcome.summary["tokens"] == tuple(names)
    run_sequence(G, n, ops)
    assert EPI_PRIMARY in G.nodes[n]

def test_run_sequence_triggers_dnfr_hook_every_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    G, node = create_nfr("dnfr")
    ops = [Emission(), Reception(), Coherence()]
    call_counter = {"count": 0}

    def stub(graph: nx.Graph, *, n_jobs: int | None = None) -> None:  # pragma: no cover - signature only
        del n_jobs
        call_counter["count"] += 1

    monkeypatch.setattr("tnfr.structural.validate_sequence", _outcome_stub)
    G.graph["compute_delta_nfr"] = stub

    run_sequence(G, node, ops)

    assert call_counter["count"] == len(ops)

def test_invalid_sequence() -> None:
    ops = [Reception(), Coherence(), Silence()]
    names = [op.name for op in ops]
    outcome = validate_sequence(names)
    assert not outcome.passed
    assert "missing" in outcome.summary["message"]
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
    outcome = validate_sequence(names)
    assert not outcome.passed
    assert operator_display_name(SELF_ORGANIZATION) in outcome.summary["message"]

def test_validate_sequence_rejects_unknown_tokens() -> None:
    names = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        RESONANCE,
        SILENCE,
        "unknown",
    ]
    outcome = validate_sequence(names)
    assert not outcome.passed
    assert "unknown tokens" in outcome.summary["message"]

def test_validate_sequence_requires_names_argument() -> None:
    with pytest.raises(TypeError) as excinfo:
        validate_sequence()

    assert "missing required argument" in str(excinfo.value)

def test_validate_sequence_rejects_empty_sequence() -> None:
    outcome = validate_sequence([])

    assert not outcome.passed
    assert outcome.summary["message"] == "empty sequence"

def test_validate_sequence_requires_string_tokens() -> None:
    outcome = validate_sequence([EMISSION, RECEPTION, 101])

    assert not outcome.passed
    assert outcome.summary["message"] == "tokens must be str"

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
    outcome = validate_sequence(names)
    assert outcome.passed, outcome.summary["message"]

def test_sequence_rejects_trailing_tokens() -> None:
    names = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        RESONANCE,
        SILENCE,
        EMISSION,
    ]
    outcome = validate_sequence(names)
    assert not outcome.passed

def test_sequence_accepts_english_tokens() -> None:
    names = [
        EMISSION,
        RECEPTION,
        COHERENCE,
        RESONANCE,
        SILENCE,
    ]
    outcome = validate_sequence(names)
    assert outcome.passed, outcome.summary["message"]

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

def test_operator_requires_glyph_assignment(graph_canon) -> None:
    class GlyphlessOperator(Operator):
        glyph = None

    op = GlyphlessOperator()
    graph = graph_canon()
    target = "node"
    graph.add_node(target)

    with pytest.raises(NotImplementedError) as excinfo:
        op(graph, target)

    assert str(excinfo.value) == "Operator without assigned glyph"

def test_create_math_nfr_records_metrics(structural_tolerances: dict[str, float]) -> None:
    G, node = create_math_nfr(
        "math-node",
        epi=0.42,
        vf=1.15,
        theta=0.12,
        dimension=3,
    )

    math_cfg = G.graph["MATH_ENGINE"]
    metrics = G.nodes[node]["math_metrics"]
    summary = G.nodes[node]["math_summary"]
    context = G.nodes[node]["math_context"]

    assert math_cfg["enabled"] is True
    assert isinstance(math_cfg["validator"], NFRValidator)
    assert isinstance(math_cfg["state_projector"], BasicStateProjector)
    assert metrics["normalized"] is True
    assert metrics["coherence_passed"] is True
    assert metrics["frequency_passed"] is True

    hilbert_space = math_cfg["hilbert_space"]
    state_projector = math_cfg["state_projector"]
    state = state_projector(
        epi=G.nodes[node][EPI_PRIMARY],
        nu_f=G.nodes[node][VF_PRIMARY],
        theta=G.nodes[node][THETA_PRIMARY],
        dim=hilbert_space.dimension,
    )
    outcome = math_cfg["validator"].validate(
        state,
        enforce_frequency_positivity=True,
    )

    coherence_summary = outcome.summary["coherence"]
    frequency_summary = outcome.summary["frequency"]
    norm_expected = float(hilbert_space.norm(state))

    assert metrics["norm"] == pytest.approx(
        norm_expected,
        rel=structural_tolerances["rtol"],
        abs=structural_tolerances["atol"],
    )
    assert metrics["coherence_value"] == pytest.approx(
        float(coherence_summary["value"]),
        rel=structural_tolerances["rtol"],
        abs=structural_tolerances["atol"],
    )
    assert metrics["coherence_threshold"] == pytest.approx(
        float(coherence_summary["threshold"]),
        rel=structural_tolerances["rtol"],
        abs=structural_tolerances["atol"],
    )
    assert metrics["frequency_value"] == pytest.approx(
        float(frequency_summary["value"]),
        rel=structural_tolerances["rtol"],
        abs=structural_tolerances["atol"],
    )
    assert metrics["frequency_spectrum_min"] == pytest.approx(
        float(frequency_summary["spectrum_min"]),
        rel=structural_tolerances["rtol"],
        abs=structural_tolerances["atol"],
    )
    assert summary["coherence"]["passed"] is True
    assert summary["frequency"]["passed"] is True
    assert context["hilbert_space"] is hilbert_space
    assert context["coherence_operator"] is math_cfg["coherence_operator"]
    assert context["frequency_operator"] is math_cfg["frequency_operator"]
    assert context["coherence_threshold"] == metrics["coherence_threshold"]

def test_create_math_nfr_merges_existing_math_config() -> None:
    base_graph = nx.Graph()
    base_graph.graph["MATH_ENGINE"] = {"enabled": False, "custom": "keep"}

    G, node = create_math_nfr("with-existing", graph=base_graph)

    assert node in G
    math_cfg = G.graph["MATH_ENGINE"]
    assert math_cfg["enabled"] is True
    assert math_cfg["custom"] == "keep"
