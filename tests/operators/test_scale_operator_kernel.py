"""Parity and provenance tests for the shared VAL/NUL proposal kernel."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.operators import apply_glyph
from tnfr.operators._scale_operator_kernel import propose_scale_operator
from tnfr.operators.metrics_structural import contraction_metrics
from tnfr.types import Glyph, real_scalar_epi


def _graph(
    *,
    glyph: Glyph,
    epi: float,
    vf: float = 2.0,
    dnfr: float = 0.3,
    factor: float,
    edge_aware: bool = True,
    epi_min: float = -1.0,
    epi_max: float = 1.0,
    clip_mode: str = "hard",
) -> nx.Graph:
    key = "VAL_scale" if glyph is Glyph.VAL else "NUL_scale"
    graph = nx.Graph(
        GLYPH_FACTORS={key: factor},
        EDGE_AWARE_ENABLED=edge_aware,
        EPI_MIN=epi_min,
        EPI_MAX=epi_max,
        CLIP_MODE=clip_mode,
    )
    graph.add_node(
        "target",
        **{
            ALIAS_EPI[0]: epi,
            ALIAS_VF[0]: vf,
            ALIAS_DNFR[0]: dnfr,
            ALIAS_THETA[0]: 0.0,
            "glyph_history": ["AL"],
        },
    )
    return graph


@pytest.mark.parametrize(
    (
        "glyph",
        "epi",
        "factor",
        "edge_aware",
        "epi_min",
        "epi_max",
        "clip_mode",
    ),
    [
        (Glyph.VAL, 0.4, 1.2, True, -1.0, 1.0, "hard"),
        (Glyph.VAL, 0.9, 2.0, True, -1.0, 1.0, "hard"),
        (Glyph.VAL, -0.6, 2.0, True, -0.7, 1.2, "hard"),
        (Glyph.VAL, 0.95, 1.02, True, -1.0, 1.0, "soft"),
        (Glyph.NUL, 0.4, 0.5, True, -1.0, 1.0, "hard"),
        (Glyph.NUL, 0.999, 0.99, True, -1.0, 1.0, "soft"),
        (Glyph.NUL, -0.4, 0.75, True, -0.8, 1.0, "hard"),
        (Glyph.VAL, 0.4, 1.2, False, -1.0, 1.0, "hard"),
        (Glyph.NUL, 0.4, 0.5, False, -1.0, 1.0, "hard"),
    ],
)
def test_pure_proposal_matches_direct_glyph_binary64_state(
    glyph,
    epi,
    factor,
    edge_aware,
    epi_min,
    epi_max,
    clip_mode,
) -> None:
    graph = _graph(
        glyph=glyph,
        epi=epi,
        factor=factor,
        edge_aware=edge_aware,
        epi_min=epi_min,
        epi_max=epi_max,
        clip_mode=clip_mode,
    )
    proposal = propose_scale_operator(
        glyph=glyph,
        factor=factor,
        vf_before=2.0,
        dnfr_before=0.3 if glyph is Glyph.NUL else None,
        configured_densification_factor=(
            1.0 / factor if glyph is Glyph.NUL else None
        ),
        edge_aware_enabled=edge_aware,
        epi_before=epi if edge_aware else None,
        epi_min=epi_min,
        epi_max=epi_max,
        epsilon=1e-12,
        clip_mode=clip_mode,
    )

    apply_glyph(graph, "target", glyph)

    assert float(get_attr(graph.nodes["target"], ALIAS_VF)) == proposal.vf_after
    observed_epi = real_scalar_epi(get_attr(graph.nodes["target"], ALIAS_EPI))
    if proposal.write_epi:
        assert observed_epi == proposal.epi_after
    else:
        assert observed_epi == epi
    if glyph is Glyph.NUL:
        assert float(get_attr(graph.nodes["target"], ALIAS_DNFR)) == (
            proposal.dnfr_after
        )
        event = graph.graph["nul_densification_log"][-1]
        assert event["node"] == "target"
        assert event["dnfr_before"] == proposal.dnfr_before
        assert event["dnfr_after"] == proposal.dnfr_after
        assert event["densification_factor"] == proposal.densification_factor
        assert event["binary64_inverse_product_residual"] == (
            proposal.binary64_inverse_product_residual
        )
    if proposal.edge_aware_adapted:
        event = graph.graph["edge_aware_interventions"][-1]
        assert event["node"] == "target"
        assert event["scale_effective"] == proposal.effective_epi_scale


def test_disabled_edge_branch_does_not_consume_boundary_inputs() -> None:
    proposal = propose_scale_operator(
        glyph=Glyph.VAL,
        factor=1.1,
        vf_before=2.0,
        edge_aware_enabled=False,
        epi_before=object(),
        epi_min="invalid",
        epi_max=None,
        epsilon="invalid",
        clip_mode=object(),
    )

    assert proposal.vf_after == pytest.approx(2.2)
    assert proposal.write_epi is False
    assert proposal.epi_after is None


def test_pure_kernel_rejects_boolean_epi_like_public_scalar_boundary() -> None:
    with pytest.raises(TNFRValueError, match="not boolean"):
        propose_scale_operator(
            glyph=Glyph.VAL,
            factor=1.1,
            vf_before=2.0,
            edge_aware_enabled=True,
            epi_before=True,
        )


def test_nul_proposal_exposes_nodal_drive_coefficient_identity() -> None:
    proposal = propose_scale_operator(
        glyph=Glyph.NUL,
        factor=0.37,
        vf_before=2.5,
        dnfr_before=-0.8,
        configured_densification_factor=1.0 / 0.37,
        edge_aware_enabled=False,
    )

    assert proposal.densification_factor is not None
    assert proposal.dnfr_after is not None
    assert proposal.binary64_inverse_product_residual == (
        proposal.requested_scale * proposal.densification_factor - 1.0
    )
    assert proposal.vf_after * proposal.dnfr_after == pytest.approx(
        proposal.vf_before * proposal.dnfr_before
    )


def test_scale_proposal_is_frozen() -> None:
    proposal = propose_scale_operator(
        glyph=Glyph.VAL,
        factor=1.1,
        vf_before=2.0,
        edge_aware_enabled=False,
    )

    with pytest.raises(FrozenInstanceError):
        proposal.vf_after = 3.0


def test_contraction_metrics_select_matching_node_event_and_magnitude() -> None:
    graph = nx.path_graph(("first", "second"))
    graph.nodes["first"].update(
        {
            ALIAS_EPI[0]: 0.4,
            ALIAS_VF[0]: 0.5,
            ALIAS_DNFR[0]: -0.6,
        }
    )
    graph.nodes["second"].update(
        {
            ALIAS_EPI[0]: 0.2,
            ALIAS_VF[0]: 0.5,
            ALIAS_DNFR[0]: 0.5,
        }
    )
    graph.graph["nul_densification_log"] = [
        {
            "node": "first",
            "dnfr_before": -0.3,
            "dnfr_after": -0.6,
            "densification_factor": 2.0,
        },
        {
            "node": "second",
            "dnfr_before": 0.25,
            "dnfr_after": 0.5,
            "densification_factor": 2.0,
        },
    ]

    metrics = contraction_metrics(graph, "first", 1.0, 0.8)

    assert metrics["dnfr_before"] == -0.3
    assert metrics["dnfr_increase"] == pytest.approx(0.3)
    assert metrics["dnfr_signed_change"] == pytest.approx(-0.3)
    assert metrics["dnfr_densified"] is True


def test_zero_pressure_contraction_is_not_reported_as_densification() -> None:
    graph = nx.Graph()
    graph.add_node(
        "target",
        **{
            ALIAS_EPI[0]: 0.4,
            ALIAS_VF[0]: 0.5,
            ALIAS_DNFR[0]: 0.0,
        },
    )
    graph.graph["nul_densification_log"] = [
        {
            "node": "target",
            "dnfr_before": 0.0,
            "dnfr_after": 0.0,
            "densification_factor": 2.0,
        }
    ]

    metrics = contraction_metrics(graph, "target", 1.0, 0.8)

    assert metrics["dnfr_densified"] is False
    assert metrics["dnfr_increase"] == 0.0
    assert metrics["densification_ratio"] == 1.0
