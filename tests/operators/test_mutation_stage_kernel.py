"""Pure ZHIR phase proposals remain immutable and match the runtime handler."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import logging
import math

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.glyph_history import push_glyph
from tnfr.operators import GLYPH_OPERATIONS
from tnfr.operators._mutation_stage_kernel import (
    commit_mutation_lifecycle,
    commit_mutation_structure,
    emit_mutation_lifecycle_log,
    propose_mutation_network_stage,
    propose_mutation_stage,
)
from tnfr.types import Glyph


class _Node:
    def __init__(self, *, theta: float, dnfr: float) -> None:
        self.theta = theta
        self.dnfr = dnfr
        self.graph: dict[str, object] = {}
        self.storage: dict[str, object] = {"glyph_history": ["IL", "OZ"]}

    def _glyph_storage(self) -> dict[str, object]:
        return self.storage


@pytest.mark.parametrize(
    ("theta", "dnfr", "factor"),
    [
        (1.45, 0.2, 0.4),
        (0.1, -0.2, 0.4),
        (4.0 * math.tau + 0.25, -0.0, 0.75),
    ],
)
def test_dynamic_proposal_matches_handler_and_complete_telemetry(
    theta: float,
    dnfr: float,
    factor: float,
) -> None:
    proposal = propose_mutation_stage(
        theta,
        dnfr,
        theta_shift_factor=factor,
    )
    node = _Node(theta=theta, dnfr=dnfr)
    history_before = list(node.storage["glyph_history"])

    GLYPH_OPERATIONS[Glyph.ZHIR](
        node,
        {"ZHIR_theta_shift_factor": factor},
    )

    assert node.theta == proposal.theta_after
    assert node.storage["glyph_history"] == history_before
    assert {
        key: node.storage[key] for key, _ in proposal.telemetry_items
    } == dict(proposal.telemetry_items)
    assert tuple(key for key, _ in proposal.telemetry_items) == (
        "_zhir_theta_shift",
        "_zhir_theta_before",
        "_zhir_theta_after",
        "_zhir_regime_changed",
        "_zhir_regime_before",
        "_zhir_regime_after",
        "_zhir_fixed_mode",
    )


def test_fixed_proposal_matches_handler_without_reading_dynamic_inputs() -> None:
    shift = 1.0e308
    proposal = propose_mutation_stage(
        -3.0 * math.tau + 0.25,
        float("nan"),
        theta_shift_factor=float("nan"),
        fixed_shift=shift,
    )
    node = _Node(theta=-3.0 * math.tau + 0.25, dnfr=float("nan"))
    node.storage.update(
        _zhir_theta_before=9.0,
        _zhir_regime_before=3,
    )

    GLYPH_OPERATIONS[Glyph.ZHIR](node, {"ZHIR_theta_shift": shift})

    assert node.theta == proposal.theta_after
    assert dict(proposal.telemetry_items) == {
        "_zhir_theta_shift": shift,
        "_zhir_fixed_mode": True,
    }
    assert node.storage["_zhir_theta_shift"] == shift
    assert node.storage["_zhir_fixed_mode"] is True
    assert node.storage["_zhir_theta_before"] == 9.0
    assert node.storage["_zhir_regime_before"] == 3


def test_mutation_proposal_is_frozen_replayable_and_rng_free() -> None:
    first = propose_mutation_stage(0.4, 0.3, theta_shift_factor=0.5)
    second = propose_mutation_stage(0.4, 0.3, theta_shift_factor=0.5)

    assert first == second
    assert first.telemetry_items == second.telemetry_items
    with pytest.raises(FrozenInstanceError):
        first.theta_after = 0.0  # type: ignore[misc]



def test_network_proposal_defers_context_log_until_postcommit_publish(
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph = nx.Graph(ZHIR_THRESHOLD_XI=0.1)
    graph.add_node(
        0,
        EPI=0.25,
        nu_f=1.0,
        DeltaNFR=0.2,
        theta=0.1,
        EPI_kind="wave",
        epi_history=[0.0, 0.1, 0.25],
        glyph_history=["IL", "OZ"],
    )
    before = deepcopy((dict(graph.nodes[0]), dict(graph.graph)))
    caplog.set_level(
        logging.INFO, logger="tnfr.operators.preconditions.mutation"
    )

    proposal = propose_mutation_network_stage(
        graph,
        0,
        {"ZHIR_theta_shift_factor": 0.5},
        tau=1.0,
    )

    assert (dict(graph.nodes[0]), dict(graph.graph)) == before
    assert not any(
        "ZHIR enabled by destabilizer" in record.getMessage()
        for record in caplog.records
    )

    commit_mutation_structure(graph, proposal)
    push_glyph(graph.nodes[0], "ZHIR", 10)
    commit_mutation_lifecycle(graph, proposal)
    assert not any(
        "ZHIR enabled by destabilizer" in record.getMessage()
        for record in caplog.records
    )

    emit_mutation_lifecycle_log(proposal)

    context_logs = [
        record
        for record in caplog.records
        if "ZHIR enabled by destabilizer" in record.getMessage()
    ]
    assert len(context_logs) == 1
    assert "distance 1" in context_logs[0].getMessage()

@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"theta": float("nan"), "delta_nfr": 0.2}, "phase state"),
        ({"theta": 0.1, "delta_nfr": float("inf")}, "DeltaNFR state"),
        (
            {
                "theta": 0.1,
                "delta_nfr": 0.2,
                "theta_shift_factor": float("inf"),
            },
            "theta shift factor",
        ),
        (
            {
                "theta": 0.1,
                "delta_nfr": 0.2,
                "fixed_shift": float("inf"),
            },
            "phase shift",
        ),
    ],
)
def test_mutation_proposal_rejects_nonfinite_active_inputs(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(TNFRValueError, match=message):
        propose_mutation_stage(**kwargs)
