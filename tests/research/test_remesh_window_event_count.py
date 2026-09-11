"""Applied-event accounting for the REMESH window diagnostic."""

from __future__ import annotations

from tnfr.riemann.remesh_window_type_signature import (
    _is_strict_positive_integer_payload,
    _run_remesh_bracket,
)


def test_insufficient_history_attempts_are_not_counted_as_events() -> None:
    (
        epi_bracket,
        nodes,
        integer_reads,
        total_reads,
        applied_events,
    ) = _run_remesh_bracket(
        n_nodes=3,
        seed=17,
        warmup_steps=1,
        tau_l_base=1,
        tau_g_base=1,
        n_events=1,
    )

    assert epi_bracket.shape == (3, 3)
    assert len(nodes) == 3
    assert integer_reads == total_reads == 6
    assert applied_events == 0


def test_window_storage_probe_matches_strict_runtime_integer_domain() -> None:
    assert _is_strict_positive_integer_payload(1)
    for value in (0, -1, True, 1.0, "1", None):
        assert not _is_strict_positive_integer_payload(value)