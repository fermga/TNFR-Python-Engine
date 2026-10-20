"""CLI text and stored-state admission precedes coercion and execution."""

from argparse import ArgumentTypeError, Namespace
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.cli import execution, main
from tnfr.cli.utils import finite_real_argument


@pytest.mark.parametrize(
    "name",
    [
        "zhir_requires_oz_window",
        "zhir_dnfr_min",
        "thol_min_len",
        "thol_max_len",
        "thol_close_dnfr",
        "si_high",
    ],
)
def test_retired_grammar_controls_reject_instead_of_accepting_ineffective_input(
    monkeypatch, capsys, name
):
    def unexpected_execution(*args, **kwargs):
        pytest.fail("retired grammar controls must reject before execution")

    monkeypatch.setattr(execution, "_run_cli_program", unexpected_execution)
    with pytest.raises(SystemExit) as failure:
        main(["run", f"--grammar.{name}=1"])
    assert failure.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command,flag",
    [
        ("network", "--probability"),
        ("run", "--dt"),
        ("run", "--p"),
        ("run", "--gamma-beta"),
        ("run", "--stop-early-fraction"),
        ("math.run", "--math-spectral-expectation-spectrum"),
        ("math.run", "--math-frequency-diagonal"),
        ("math.run", "--math-generator-diagonal"),
        ("epi.validate", "--tolerance"),
    ],
)
@pytest.mark.parametrize("token", ["1e-400", "-1e-400", "NaN", "1e400"])
def test_scalar_flags_reject_before_running_or_writing(
    monkeypatch, tmp_path, capsys, command, flag, token
):
    def unexpected_execution(*args, **kwargs):
        pytest.fail("invalid text must reject before execution")

    monkeypatch.setattr(execution, "_run_cli_program", unexpected_execution)
    monkeypatch.setattr("tnfr.sdk.run_study", unexpected_execution)
    output = tmp_path / "existing.json"
    output.write_bytes(b"retained evidence\n")
    arguments = [command, f"{flag}={token}"]
    if command == "network":
        arguments.extend(["--output", str(output)])
    with pytest.raises(SystemExit) as failure:
        main(arguments)
    assert failure.value.code == 2
    assert capsys.readouterr().out == ""
    assert output.read_bytes() == b"retained evidence\n"


@pytest.mark.parametrize(
    "token,expected",
    [
        ("0e-9999", 0.0),
        ("-0", 0.0),
        (".5", 0.5),
        ("+1.25", 1.25),
        ("5e-324", np.nextafter(0.0, 1.0)),
    ],
)
def test_decimal_flags_preserve_valid_zero_rounding_and_subnormal(token, expected):
    assert finite_real_argument(token) == expected


@pytest.mark.parametrize("token", ["true", "1/2", "not-a-number", True])
def test_scalar_text_does_not_admit_boolean_or_new_fraction_syntax(token):
    with pytest.raises(ArgumentTypeError):
        finite_real_argument(token)


@pytest.mark.parametrize(
    "command,flag",
    [
        ("network", "--probability"),
        ("run", "--p"),
        ("run", "--stop-early-fraction"),
        ("math.run", "--stop-early-fraction"),
    ],
)
def test_probability_text_cannot_round_into_the_admitted_interval(command, flag):
    with pytest.raises(SystemExit) as failure:
        main([command, f"{flag}=1.0000000000000000000001"])
    assert failure.value.code == 2


def _math_graph():
    graph = nx.Graph()
    graph.add_node(0)
    graph.graph["MATH_ENGINE"] = execution._build_math_engine_config(
        graph, Namespace(math_dimension=2)
    )
    return graph


@pytest.mark.parametrize(
    "aliases",
    [execution.EPI_ALIAS_KEYS, execution.VF_ALIAS_KEYS, execution.THETA_ALIAS_KEYS],
)
@pytest.mark.parametrize("invalid", [None, True, "0.5", Fraction(1, 10**400)])
def test_math_summary_cannot_coerce_primary_state_or_fall_back(aliases, invalid):
    graph = _math_graph()
    graph.nodes[0].update({aliases[0]: invalid, aliases[1]: 0.5})
    with pytest.raises((TypeError, ValueError)):
        execution._log_math_engine_summary(graph)


def test_math_summary_passes_legacy_phase_and_signed_form_to_projector():
    graph = _math_graph()
    graph.nodes[0].update(
        {
            execution.EPI_ALIAS_KEYS[-1]: -0.25,
            execution.VF_ALIAS_KEYS[-1]: 0.0,
            execution.THETA_ALIAS_KEYS[-1]: 0.75,
        }
    )
    config = graph.graph["MATH_ENGINE"]
    original = config["state_projector"]
    observed = []

    def projector(**values):
        observed.append(values)
        return original(**values)

    config["state_projector"] = projector
    execution._log_math_engine_summary(graph)
    assert observed == [{"epi": -0.25, "nu_f": 0.0, "theta": 0.75, "dim": 2}]


@pytest.mark.parametrize("invalid", [True, "1", Fraction(1, 10**400), [1]])
def test_namespace_spectral_array_does_not_bypass_scalar_admission(invalid):
    with pytest.raises(ValueError):
        execution._to_float_array([invalid], name="test spectrum")


@pytest.mark.parametrize("topology", ["ring", "complete", "erdos"])
@pytest.mark.parametrize(
    "field,value", [("nodes", True), ("nodes", 2.5), ("seed", True), ("seed", "1")]
)
def test_programmatic_topology_uses_original_integer_admission(topology, field, value):
    args = dict(nodes=3, topology=topology, p=0.5, seed=1)
    args[field] = value
    with pytest.raises(ValueError):
        execution.build_basic_graph(Namespace(**args))


@pytest.mark.parametrize(
    "value", [True, Fraction(1, 10**400), Fraction(1) + Fraction(1, 10**400)]
)
def test_programmatic_probability_matches_sdk_and_parser_domain(value):
    with pytest.raises(ValueError):
        execution.build_basic_graph(
            Namespace(nodes=3, topology="erdos", p=value, seed=1)
        )


@pytest.mark.parametrize(
    "options",
    [
        {"dt": True},
        {"dt": Fraction(1, 10**400)},
        {"glyph_hysteresis_window": 1.5},
        {"um_candidate_count": True},
        {"um_candidate_count": 1.5},
        {"stop_early_window": True},
        {"stop_early_window": 1.5},
        {"stop_early_fraction": True},
        {"stop_early_fraction": Fraction(1, 10**400)},
        {"stop_early_fraction": Fraction(1) + Fraction(1, 10**400)},
        {"selector": "unknown"},
        {"grammar_canon": "unknown"},
    ],
)
def test_programmatic_policy_does_not_hide_invalid_source_values(options):
    with pytest.raises((TypeError, ValueError)):
        execution.apply_cli_config(nx.Graph(), Namespace(config=None, **options))


def test_cli_valid_policies_survive_json_readback_with_the_same_runtime_meaning():
    from tnfr.dynamics.runtime import _resolve_early_stop
    from tnfr.utils.io import json_dumps, json_loads

    graph = nx.Graph()
    execution.apply_cli_config(
        graph,
        Namespace(
            config=None,
            dt=Fraction(1, 8),
            grammar_canon="false",
            stop_early_window=np.int64(3),
            stop_early_fraction=Fraction(3, 4),
        ),
    )
    restored = nx.Graph()
    restored.graph.update(json_loads(json_dumps(graph.graph)))
    assert graph.graph["DT"] == restored.graph["DT"] == 0.125
    assert graph.graph["GRAMMAR_CANON"]["enabled"] is False
    assert restored.graph["GRAMMAR_CANON"]["enabled"] is False
    assert _resolve_early_stop(graph) == _resolve_early_stop(restored) == (3, 0.75)


def test_math_summary_rebuilds_threshold_instead_of_trusting_cached_validator(
    monkeypatch,
):
    graph = _math_graph()
    config = graph.graph["MATH_ENGINE"]
    config["spectral_expectation_threshold"] = 100.0
    config["coherence_threshold"] = 100.0
    messages = []
    monkeypatch.setattr(
        execution.logger, "info", lambda fmt, *args: messages.append(fmt % args)
    )

    execution._log_math_engine_summary(graph)

    assert any(
        "threshold passed=False" in text and "floor=100.000000" in text
        for text in messages
    )


@pytest.mark.parametrize("value", [True, Fraction(1, 10**400), float("nan")])
def test_cached_math_validator_cannot_hide_invalid_current_threshold(value):
    graph = _math_graph()
    config = graph.graph["MATH_ENGINE"]
    config["spectral_expectation_threshold"] = value
    config["coherence_threshold"] = value
    with pytest.raises(ValueError):
        execution._log_math_engine_summary(graph)


def test_disabled_math_flag_uses_boolean_parser_without_building_engine(monkeypatch):
    def unexpected_engine(*args, **kwargs):
        pytest.fail("explicit false flag must not build a math engine")

    monkeypatch.setattr(execution, "_build_math_engine_config", unexpected_engine)
    graph = nx.Graph()
    execution._configure_math_engine(graph, Namespace(math_engine="false"))
    assert "MATH_ENGINE" not in graph.graph
