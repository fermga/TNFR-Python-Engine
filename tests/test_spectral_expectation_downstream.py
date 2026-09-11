"""Downstream contracts for the auxiliary spectral-expectation vocabulary."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from types import SimpleNamespace

import networkx as nx
import pytest

np = pytest.importorskip("numpy")

from tnfr.cli.arguments import _add_math_run_parser
from tnfr.cli.execution import _build_math_engine_config, _log_math_engine_summary
from tnfr.constants import EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from tnfr.dynamics.runtime import _advance_math_engine
from tnfr.errors import TNFRValueError
from tnfr.mathematics import HilbertSpace, SpectralExpectationOperator
from tnfr.node import NodeNX
from tnfr.structural import create_math_nfr, create_nfr


def _assert_auxiliary_payload(payload: dict[str, object]) -> None:
    assert payload["value"] == pytest.approx(4.0)
    assert payload["threshold"] == pytest.approx(3.0)
    assert payload["passed"] is True
    assert payload["metric_kind"] == "spectral_operator_expectation"
    assert payload["range"] == "unbounded_real"
    assert payload["bounded"] is False
    assert payload["canonical_coherence"] is False
    assert payload["canonical_coherence_certified"] is False
    assert payload["records_to_C_steps"] is False
    assert isinstance(payload["provenance"], str)


def _math_namespace(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "math_dimension": 2,
        "math_spectral_expectation_spectrum": [4.0, 4.0],
        "math_spectral_expectation_floor": None,
        "math_spectral_expectation_threshold": 3.0,
        "math_frequency_diagonal": None,
        "math_generator_diagonal": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _parse_math_args(options: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    _add_math_run_parser(subparsers)
    return parser.parse_args(["math.run", "--nodes", "2", *options])


def test_create_math_nfr_reports_unbounded_spectral_expectation() -> None:
    graph, node = create_math_nfr(
        "spectral-seed",
        epi=0.4,
        vf=1.2,
        theta=0.05,
        dimension=2,
        spectral_spectrum=[4.0, 4.0],
        spectral_expectation_threshold=3.0,
    )

    data = graph.nodes[node]
    payload = data["math_metrics"]["spectral_operator_expectation"]
    _assert_auxiliary_payload(payload)
    assert data["math_metrics"]["coherence_value"] == payload["value"]
    assert data["math_metrics"]["coherence_passed"] == payload["passed"]
    assert data["math_summary"]["coherence"] is payload
    assert data["math_summary"]["spectral_operator_expectation"] is payload

    context = data["math_context"]
    config = graph.graph["MATH_ENGINE"]
    assert context["coherence_operator"] is context["spectral_operator"]
    assert config["coherence_operator"] is config["spectral_operator"]
    assert context["canonical_coherence_certified"] is False
    assert config["records_to_C_steps"] is False
    assert "C_steps" not in graph.graph


def test_create_math_nfr_legacy_keywords_materialize_canonical_fields() -> None:
    graph, node = create_math_nfr(
        "legacy-spectral-seed",
        dimension=2,
        coherence_spectrum=[4.0, 4.0],
        coherence_threshold=3.0,
    )

    config = graph.graph["MATH_ENGINE"]
    payload = graph.nodes[node]["math_summary"]["spectral_operator_expectation"]
    _assert_auxiliary_payload(payload)
    assert config["spectral_expectation_threshold"] == pytest.approx(3.0)
    assert config["coherence_threshold"] == pytest.approx(3.0)


def test_node_metrics_expose_canonical_payload_and_compatibility_aliases() -> None:
    graph, node = create_nfr("node", epi=0.4, vf=1.2, theta=0.05)
    operator = SpectralExpectationOperator([4.0, 4.0])
    adapter = NodeNX(
        graph,
        node,
        spectral_operator=operator,
        spectral_expectation_threshold=3.0,
    )

    assert adapter.hilbert_space.dimension == 2
    result = adapter.run_sequence_with_validation([], enable_validation=False)

    for label in ("pre_metrics", "post_metrics"):
        payload = result[label]["spectral_operator_expectation"]
        _assert_auxiliary_payload(payload)
        assert result[label]["coherence"] is True
        assert result[label]["coherence_expectation"] == payload["value"]
        assert result[label]["canonical_coherence_certified"] is False
    assert adapter.coherence_operator is adapter.spectral_operator
    assert adapter.coherence_threshold == adapter.spectral_expectation_threshold


def test_node_historical_constructor_and_method_keywords_remain_aliases() -> None:
    graph, node = create_nfr("legacy-node", epi=0.4, vf=1.2, theta=0.05)
    operator = SpectralExpectationOperator([4.0, 4.0])
    adapter = NodeNX(
        graph,
        node,
        coherence_operator=operator,
        coherence_threshold=3.0,
    )

    assert adapter.spectral_operator is operator
    assert adapter.spectral_expectation_threshold == pytest.approx(3.0)
    assert adapter.hilbert_space.dimension == 2

    plain_graph, plain_node = create_nfr(
        "legacy-method-node", epi=0.4, vf=1.2, theta=0.05
    )
    plain_adapter = NodeNX(plain_graph, plain_node)
    result = plain_adapter.run_sequence_with_validation(
        [],
        coherence_operator=operator,
        coherence_threshold=3.0,
        enable_validation=False,
    )

    _assert_auxiliary_payload(
        result["post_metrics"]["spectral_operator_expectation"]
    )


def test_runtime_materializes_canonical_history_from_legacy_config() -> None:
    graph, _ = create_math_nfr(
        "runtime-node",
        dimension=2,
        spectral_spectrum=[4.0, 4.0],
        spectral_expectation_threshold=3.0,
    )
    config = graph.graph["MATH_ENGINE"]
    config.pop("spectral_operator")
    config.pop("spectral_expectation_threshold")
    history: dict[str, list[object]] = {}

    _advance_math_engine(graph, dt=0.1, step_idx=7, hist=history)

    payload = history["math_engine_summary"][0][
        "spectral_operator_expectation"
    ]
    _assert_auxiliary_payload(payload)
    assert history["math_engine_summary"][0]["coherence"] is payload
    assert history["math_engine_coherence"] == [payload["value"]]
    assert history["math_engine_spectral_operator_expectation"] == [
        payload["value"]
    ]
    assert config["coherence_operator"] is config["spectral_operator"]
    assert "C_steps" not in history


def test_cli_accepts_canonical_and_historical_flag_spellings() -> None:
    canonical = _parse_math_args(
        [
            "--math-spectral-expectation-spectrum",
            "4",
            "4",
            "--math-spectral-expectation-threshold",
            "3",
        ]
    )
    legacy = _parse_math_args(
        [
            "--math-coherence-spectrum",
            "4",
            "4",
            "--math-coherence-threshold",
            "3",
        ]
    )

    assert canonical.math_spectral_expectation_spectrum == [4.0, 4.0]
    assert legacy.math_spectral_expectation_spectrum == [4.0, 4.0]
    assert canonical.math_spectral_expectation_threshold == pytest.approx(3.0)
    assert legacy.math_spectral_expectation_threshold == pytest.approx(3.0)

    graph = nx.Graph()
    graph.add_nodes_from((0, 1))
    canonical_config = _build_math_engine_config(graph, canonical)
    legacy_config = _build_math_engine_config(graph, legacy)
    for config in (canonical_config, legacy_config):
        assert config["spectral_expectation_threshold"] == pytest.approx(3.0)
        assert config["coherence_threshold"] == pytest.approx(3.0)
        assert config["bounded"] is False
        assert config["canonical_coherence_certified"] is False
        assert config["records_to_C_steps"] is False
        assert np.allclose(config["spectral_operator"].spectrum(), [4.0, 4.0])


def test_cli_summary_names_the_auxiliary_metric(
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph = nx.Graph()
    graph.add_node(
        "node",
        **{EPI_PRIMARY: 0.4, VF_PRIMARY: 1.2, THETA_PRIMARY: 0.05},
    )
    graph.graph["MATH_ENGINE"] = _build_math_engine_config(
        graph,
        _math_namespace(),
    )

    with caplog.at_level(logging.INFO, logger="tnfr.cli.execution"):
        _log_math_engine_summary(graph)

    assert "Spectral expectation threshold passed=True" in caplog.text
    assert "canonical_coherence_certified=False" in caplog.text
    assert "Coherence ≥" not in caplog.text


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: create_math_nfr(
                "conflict",
                dimension=2,
                spectral_expectation_threshold=3.0,
                coherence_threshold=2.0,
            ),
            "not both",
        ),
        (
            lambda: create_math_nfr(
                "nonfinite",
                dimension=2,
                spectral_expectation_threshold=float("nan"),
            ),
            "finite",
        ),
        (
            lambda: NodeNX(
                create_nfr("bad-node")[0],
                "bad-node",
                hilbert_space=HilbertSpace(3),
                spectral_operator=SpectralExpectationOperator([1.0, 2.0]),
            ),
            "dimension",
        ),
        (
            lambda: _build_math_engine_config(
                nx.Graph([(0, 1)]),
                _math_namespace(math_spectral_expectation_threshold=float("inf")),
            ),
            "finite",
        ),
        (
            lambda: NodeNX(
                create_nfr("non-hermitian")[0],
                "non-hermitian",
                spectral_operator=SpectralExpectationOperator(
                    [[1.0, 1.0], [0.0, 1.0]],
                    ensure_hermitian=False,
                ),
            ),
            "Hermitian",
        ),
    ],
)
def test_invalid_or_contradictory_spectral_inputs_are_rejected(
    call: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises((TNFRValueError, ValueError), match=message):
        call()


def test_programmatic_legacy_cli_namespace_remains_supported() -> None:
    graph = nx.Graph()
    graph.add_nodes_from((0, 1))
    args = SimpleNamespace(
        math_dimension=2,
        math_coherence_spectrum=[4.0, 4.0],
        math_coherence_c_min=None,
        math_coherence_threshold=3.0,
        math_frequency_diagonal=None,
        math_generator_diagonal=None,
    )

    config = _build_math_engine_config(graph, args)

    assert config["spectral_expectation_threshold"] == pytest.approx(3.0)
    assert config["coherence_threshold"] == pytest.approx(3.0)