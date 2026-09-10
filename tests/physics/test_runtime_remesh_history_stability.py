"""Executed binary64 REMESH to exact companion-history bridge."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import networkx as nx
import pytest

import tnfr.physics.remesh_history_stability as exact_module
import tnfr.physics.runtime_remesh_history_stability as bridge_module
from tnfr.errors import TNFRValueError
from tnfr.operators import build_operator_event_schedule, execute_event_remesh_cycle
from tnfr.physics.runtime_remesh_history_stability import (
    RuntimeRemeshHistoryBridgeObservation,
    observe_runtime_remesh_history_bridge,
)


def _graph(
    *,
    current: tuple[float, float] = (2.0, 0.0),
    history: tuple[tuple[float, float], ...] = ((0.0, 2.0),),
    alpha: float = 0.5,
    tau_local: int = 1,
    tau_global: int = 1,
    epi_min: float = -10.0,
    epi_max: float = 10.0,
    clip_mode: str = "hard",
) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=19,
        _gamma_spec={"type": "none"},
        REMESH_TAU_GLOBAL=tau_global,
        REMESH_TAU_LOCAL=tau_local,
        REMESH_ALPHA=alpha,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=epi_min,
        EPI_MAX=epi_max,
        CLIP_MODE=clip_mode,
    )
    for node, epi in enumerate(current):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    graph.graph["_epi_hist"] = deque(
        (
            {node: value for node, value in enumerate(snapshot)}
            for snapshot in history
        ),
        maxlen=64,
    )
    return graph


def _cycle(graph: nx.Graph, *, weights=(1.0, 1.0)):
    schedule = build_operator_event_schedule(
        (),
        start_time=graph.graph["_t"],
        flow_durations=(0.0,),
    )
    return execute_event_remesh_cycle(
        graph,
        schedule,
        metric_weights=weights,
    )


def test_public_physics_facade_and_stub_expose_bridge_api() -> None:
    import tnfr.physics as physics

    assert {
        "RuntimeRemeshHistoryBridgeObservation",
        "observe_runtime_remesh_history_bridge",
    } <= set(physics.__all__)
    assert (
        physics.RuntimeRemeshHistoryBridgeObservation
        is RuntimeRemeshHistoryBridgeObservation
    )
    assert (
        physics.observe_runtime_remesh_history_bridge
        is observe_runtime_remesh_history_bridge
    )

    package = Path(physics.__file__).parent
    stub = (package / "runtime_remesh_history_stability.pyi").read_text(
        encoding="utf-8"
    )
    assert "class RuntimeRemeshHistoryBridgeObservation" in stub
    assert "def observe_runtime_remesh_history_bridge" in stub


def test_exact_representable_runtime_transition_matches_companion() -> None:
    bridge = observe_runtime_remesh_history_bridge(
        _cycle(_graph(), weights=(1.0, 3.0))
    )

    assert type(bridge) is RuntimeRemeshHistoryBridgeObservation
    assert bridge.bridge_observation_certified
    assert bridge.exact_history == (
        (Fraction(2), Fraction(0)),
        (Fraction(0), Fraction(2)),
    )
    assert bridge.exact_ideal_next_field == (
        Fraction(1, 2),
        Fraction(3, 2),
    )
    assert bridge.exact_runtime_raw_next_field == bridge.exact_ideal_next_field
    assert (
        bridge.exact_runtime_bounded_next_field
        == bridge.exact_ideal_next_field
    )
    assert bridge.exact_rounding_residual == (Fraction(0), Fraction(0))
    assert bridge.exact_clipping_residual == (Fraction(0), Fraction(0))
    assert bridge.exact_total_residual == (Fraction(0), Fraction(0))
    assert bridge.raw_binary64_replay_identified
    assert bridge.bounded_binary64_replay_identified
    assert bridge.exact_jensen_dissipation > 0
    assert bridge.lifted_runtime_raw_energy_nonincrease_observed
    assert bridge.lifted_runtime_bounded_energy_nonincrease_observed
    assert bridge.lifted_runtime_raw_energy_nonincrease_sufficiently_certified
    assert bridge.lifted_runtime_bounded_energy_nonincrease_sufficiently_certified
    assert bridge.hard_clipping_step_nonexpansive_certified
    assert bridge.lifted_runtime_barycenter_preserved_observed
    assert not bridge.live_runtime_history_advance_certified
    assert not bridge.repeated_runtime_stability_certified
    assert not bridge.companion_temporal_convergence_transferred_to_runtime
    assert not bridge.schedule_remesh_composition_certified
    assert not bridge.future_stability_certified


def test_signed_rounding_residual_can_reverse_the_ideal_energy_drop() -> None:
    current = (-0.2820072804276006, -0.884890940701563)
    delayed = (-0.28200728042760065, -0.8848909407015633)
    graph = _graph(
        current=current,
        history=(delayed,),
        alpha=0.7342090787834955,
    )

    bridge = observe_runtime_remesh_history_bridge(_cycle(graph))

    assert bridge.exact_max_abs_rounding_residual > 0
    assert bridge.exact_clipping_residual == (Fraction(0), Fraction(0))
    assert bridge.exact_total_residual == bridge.exact_rounding_residual
    assert bridge.exact_jensen_dissipation > 0
    assert bridge.exact_rounding_augmented_energy_defect > 0
    assert bridge.exact_lifted_runtime_raw_energy_drop < 0
    assert bridge.exact_lifted_runtime_bounded_energy_drop < 0
    assert not bridge.lifted_runtime_raw_energy_nonincrease_observed
    assert not bridge.lifted_runtime_bounded_energy_nonincrease_observed
    assert not bridge.lifted_runtime_raw_energy_nonincrease_sufficiently_certified
    assert not bridge.lifted_runtime_bounded_energy_nonincrease_sufficiently_certified
    assert bridge.hard_clipping_step_nonexpansive_certified


def test_hard_clipping_has_nonpositive_exact_energy_defect() -> None:
    graph = _graph(
        current=(2.0, -2.0),
        history=((2.0, -2.0),),
        epi_min=-1.0,
        epi_max=1.0,
    )

    bridge = observe_runtime_remesh_history_bridge(_cycle(graph))

    assert bridge.clipping_intervened
    assert bridge.exact_jensen_dissipation == 0
    assert bridge.exact_rounding_augmented_energy_defect == 0
    assert bridge.exact_clipping_augmented_energy_defect < 0
    assert bridge.exact_total_augmented_energy_defect < 0
    assert bridge.exact_lifted_runtime_bounded_energy_drop > 0
    assert bridge.lifted_runtime_bounded_energy_nonincrease_observed
    assert bridge.hard_clipping_step_nonexpansive_certified
    assert not bridge.lifted_runtime_barycenter_preserved_observed


def test_soft_clipping_counterexample_is_observed_without_promotion() -> None:
    current = (0.9711902890183025, 0.9711911532966678)
    graph = _graph(
        current=current,
        history=(current,),
        epi_min=-1.0,
        epi_max=1.0,
        clip_mode="soft",
    )

    bridge = observe_runtime_remesh_history_bridge(_cycle(graph))

    assert bridge.exact_jensen_dissipation == 0
    assert bridge.exact_clipping_augmented_energy_defect > 0
    assert bridge.exact_lifted_runtime_bounded_energy_drop < 0
    assert not bridge.lifted_runtime_bounded_energy_nonincrease_observed
    assert not bridge.lifted_runtime_bounded_energy_nonincrease_sufficiently_certified
    assert not bridge.hard_clipping_step_nonexpansive_certified


def test_alpha_one_uses_only_global_delay_and_preserves_pure_delay_energy() -> None:
    graph = _graph(
        current=(3.0, -1.0),
        history=((5.0, 4.0), (0.0, 2.0)),
        alpha=1.0,
        tau_local=1,
        tau_global=2,
    )

    bridge = observe_runtime_remesh_history_bridge(_cycle(graph))

    assert bridge.exact_transition.certificate.alpha_one_pure_delay_map_certified
    assert bridge.exact_ideal_next_field == (Fraction(5), Fraction(4))
    assert bridge.exact_runtime_raw_next_field == bridge.exact_ideal_next_field
    assert bridge.exact_rounding_residual == (Fraction(0), Fraction(0))
    assert bridge.exact_clipping_residual == (Fraction(0), Fraction(0))
    assert bridge.exact_jensen_dissipation == 0
    assert bridge.exact_lifted_runtime_bounded_energy_drop == 0


def test_alpha_one_drops_an_inactive_longer_local_delay_from_the_companion() -> None:
    graph = _graph(
        current=(3.0, -1.0),
        history=((9.0, 8.0), (7.0, 6.0), (0.0, 2.0)),
        alpha=1.0,
        tau_local=3,
        tau_global=1,
    )

    bridge = observe_runtime_remesh_history_bridge(_cycle(graph))

    assert bridge.exact_transition.certificate.gamma == 0
    assert bridge.exact_transition.certificate.active_max_delay == 1
    assert bridge.exact_history == (
        (Fraction(3), Fraction(-1)),
        (Fraction(0), Fraction(2)),
    )
    assert bridge.exact_ideal_next_field == (Fraction(0), Fraction(2))
    assert bridge.exact_runtime_raw_next_field == bridge.exact_ideal_next_field


def test_insufficient_history_cycle_is_rejected() -> None:
    cycle = _cycle(
        _graph(
            history=(),
            tau_local=1,
            tau_global=2,
        )
    )

    assert not cycle.remesh.applied
    with pytest.raises(TNFRValueError, match="applied REMESH"):
        observe_runtime_remesh_history_bridge(cycle)


def test_tampered_cycle_and_bridge_fail_closed() -> None:
    cycle = _cycle(_graph())
    bridge = observe_runtime_remesh_history_bridge(cycle)
    object.__setattr__(cycle, "exact_total_weighted_mean_drift", Fraction(99))

    assert not cycle._proof_fields_are_intact()
    assert not bridge.bridge_observation_certified
    assert bridge.failed_conditions == (
        "runtime_remesh_history_bridge_proof_fields_intact",
    )
    with pytest.raises(TNFRValueError, match="unsealed, tampered"):
        observe_runtime_remesh_history_bridge(cycle)


def test_direct_bridge_field_tampering_fails_closed() -> None:
    bridge = observe_runtime_remesh_history_bridge(_cycle(_graph()))
    object.__setattr__(
        bridge,
        "exact_total_residual",
        (Fraction(7), Fraction(7)),
    )

    assert not bridge.bridge_observation_certified
    assert not bridge.lifted_runtime_bounded_energy_nonincrease_observed


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("exact_jensen_dissipation", Fraction(999)),
        (
            "exact_history",
            (
                (Fraction(8), Fraction(7)),
                (Fraction(6), Fraction(5)),
            ),
        ),
        ("exact_max_abs_total_residual", Fraction(5)),
        ("clipping_intervened", True),
    ],
)
def test_privately_resealed_inconsistent_derivatives_fail_closed(
    field_name: str,
    forged_value: object,
) -> None:
    bridge = observe_runtime_remesh_history_bridge(_cycle(_graph()))
    forged = replace(
        bridge,
        **{field_name: forged_value},
        _proof_stamp=(),
    )
    resealed = bridge_module._seal(forged)

    assert not resealed.bridge_observation_certified
    assert resealed.failed_conditions == (
        "runtime_remesh_history_bridge_proof_fields_intact",
    )


def test_resealed_hostile_nested_transition_is_rejected_without_equality() -> None:
    equality_calls: list[str] = []

    class AlwaysEqual:
        def __eq__(self, other: object) -> bool:
            del other
            equality_calls.append("called")
            return True

    bridge = observe_runtime_remesh_history_bridge(_cycle(_graph()))
    forged_transition = replace(
        bridge.exact_transition,
        active_centered_fields=AlwaysEqual(),
        _proof_stamp=(),
    )
    forged_transition = exact_module._seal(
        forged_transition,
        exact_module.UniformRemeshHistoryTransitionObservation,
        exact_module._TRANSITION_PROOF_VERSION,
    )
    forged_bridge = bridge_module._seal(
        replace(
            bridge,
            exact_transition=forged_transition,
            _proof_stamp=(),
        )
    )

    assert not forged_bridge.bridge_observation_certified
    assert equality_calls == []


def test_wrong_public_input_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="EventRemeshCycleResult"):
        observe_runtime_remesh_history_bridge(object())  # type: ignore[arg-type]
