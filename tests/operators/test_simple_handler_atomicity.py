"""Finite proposals and atomic commits for the simple runtime handlers."""

from __future__ import annotations

import math
import sys
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.errors import TNFRValueError
from tnfr.operators import (
    GLYPH_OPERATIONS,
    apply_glyph,
    get_glyph_factors,
)
from tnfr.types import Glyph, real_scalar_epi


def _graph(*, factors: dict | None = None, **config) -> nx.Graph:
    graph = nx.Graph(GLYPH_FACTORS={} if factors is None else factors, **config)
    graph.add_node(
        0,
        **{
            ALIAS_EPI[0]: 0.4,
            ALIAS_VF[0]: 2.0,
            ALIAS_DNFR[0]: 0.3,
            ALIAS_D2EPI[0]: 0.2,
            ALIAS_THETA[0]: 0.1,
            "epi_kind": "seed",
            "epi_history": [0.0, 1.0],
            "glyph_history": ["AL"],
        },
    )
    return graph


def _graph_without_adapter_cache(graph: nx.Graph) -> dict:
    return {key: value for key, value in graph.graph.items() if key != "_node_cache"}


@pytest.mark.parametrize(
    "bounds",
    [
        {"EPI_MIN": float("nan")},
        {"EPI_MAX": float("inf")},
        {"EPI_MIN": 1.0, "EPI_MAX": -1.0},
    ],
)
def test_epi_bounds_reject_before_al_state_or_history_commit(bounds):
    graph = _graph(**bounds)
    node_before = deepcopy(graph.nodes[0])
    graph_before = deepcopy(graph.graph)

    with pytest.raises(TNFRValueError, match="EPI"):
        apply_glyph(graph, 0, "AL")

    assert graph.nodes[0] == node_before
    assert _graph_without_adapter_cache(graph) == graph_before


@pytest.mark.parametrize(
    ("glyph", "factors", "updates", "message"),
    [
        (
            "OZ",
            {"OZ_dnfr_factor": 1e308},
            {ALIAS_DNFR[0]: 2.0},
            "OZ DeltaNFR proposal",
        ),
        (
            "THOL",
            {"THOL_accel": 1e308},
            {ALIAS_DNFR[0]: 2.0, ALIAS_D2EPI[0]: 2.0},
            "THOL DeltaNFR contribution",
        ),
    ],
)
def test_overflowing_simple_proposal_rejects_atomically(
    glyph, factors, updates, message
):
    graph = _graph(factors=factors)
    graph.nodes[0].update(updates)
    node_before = deepcopy(graph.nodes[0])

    with pytest.raises(TNFRValueError, match=message):
        apply_glyph(graph, 0, glyph)

    assert graph.nodes[0] == node_before


@pytest.mark.parametrize("glyph", ["OZ", "NAV"])
def test_rejected_random_overflow_restores_jitter_progress(glyph):
    factor_key = "NAV_jitter" if glyph == "NAV" else None
    factors = {factor_key: sys.float_info.max} if factor_key else {}
    config = {"RANDOM_SEED": 7}
    if glyph == "OZ":
        config.update(OZ_NOISE_MODE=True, OZ_SIGMA=sys.float_info.max)
    graph = _graph(factors=factors, **config)
    node_before = deepcopy(graph.nodes[0])

    with pytest.raises(TNFRValueError, match="sample|proposal"):
        apply_glyph(graph, 0, glyph)

    assert graph.nodes[0] == node_before
    assert "_rng_jitter_progress" not in graph.nodes[0]


def test_oz_noise_rejects_nonfinite_sigma_before_draw_or_state_commit():
    graph = _graph(OZ_NOISE_MODE=True, OZ_SIGMA=float("nan"), RANDOM_SEED=7)
    node_before = deepcopy(graph.nodes[0])

    with pytest.raises(TNFRValueError, match="OZ noise sigma"):
        apply_glyph(graph, 0, "OZ")

    assert graph.nodes[0] == node_before
    assert "_rng_jitter_progress" not in graph.nodes[0]


def test_runtime_branches_do_not_read_inactive_factors():
    oz = _graph(
        factors={"OZ_dnfr_factor": 0.5},
        OZ_NOISE_MODE=True,
        OZ_SIGMA=0.1,
        RANDOM_SEED=7,
    )
    apply_glyph(oz, 0, "OZ")
    assert list(oz.nodes[0]["glyph_history"])[-1] == "OZ"

    nav = _graph(
        factors={"NAV_eta": 2.0, "NAV_jitter": 0.1},
        NAV_STRICT=True,
        NAV_RANDOM=False,
    )
    apply_glyph(nav, 0, "NAV")
    assert nav.nodes[0][ALIAS_DNFR[0]] == pytest.approx(2.1)

    zhir = _graph(
        factors={"ZHIR_theta_shift": 0.2, "ZHIR_theta_shift_factor": 0.0}
    )
    apply_glyph(zhir, 0, "ZHIR")
    assert zhir.nodes[0][ALIAS_THETA[0]] == pytest.approx(0.3)


def test_fixed_zhir_shift_is_finite_and_normalized_before_telemetry():
    graph = _graph(factors={"ZHIR_theta_shift": 1e308})

    apply_glyph(graph, 0, "ZHIR")

    theta = graph.nodes[0][ALIAS_THETA[0]]
    assert math.isfinite(theta)
    assert 0.0 <= theta < math.tau
    assert graph.nodes[0]["_zhir_theta_shift"] == 1e308
    assert graph.nodes[0]["_zhir_fixed_mode"] is True


def test_nav_noop_rejects_before_state_history_or_jitter_progress():
    graph = _graph(
        factors={"NAV_eta": 0.0, "NAV_jitter": 0.0}, NAV_RANDOM=False
    )
    graph.nodes[0][ALIAS_DNFR[0]] = 0.0
    node_before = deepcopy(graph.nodes[0])

    with pytest.raises(TNFRValueError, match="NAV must change"):
        apply_glyph(graph, 0, "NAV")

    assert graph.nodes[0] == node_before
    assert "_rng_jitter_progress" not in graph.nodes[0]


class _BareNode:
    def __init__(self, *, epi=0.2, kind="seed", neighbors=()):
        self.EPI = epi
        self.vf = 1.0
        self.dnfr = 0.2
        self.d2EPI = 0.1
        self.theta = 0.1
        self.epi_kind = kind
        self.graph = {}
        self._neighbors = list(neighbors)
        self.storage = {"glyph_history": []}

    def neighbors(self):
        return iter(self._neighbors)

    def _glyph_storage(self):
        return self.storage

    def offset(self):
        return 0


class _RejectDominantKindNode(_BareNode):
    def __init__(self, **kwargs):
        self._epi_kind = "seed"
        super().__init__(**kwargs)

    @property
    def epi_kind(self):
        return self._epi_kind

    @epi_kind.setter
    def epi_kind(self, value):
        if value == "dominant":
            raise RuntimeError("kind setter rejected the proposal")
        self._epi_kind = value


def test_en_rolls_back_epi_when_identity_commit_is_rejected():
    neighbor = _BareNode(epi=1.0, kind="dominant")
    node = _RejectDominantKindNode(epi=0.2, kind="seed", neighbors=[neighbor])
    factors = get_glyph_factors(node, Glyph.EN)

    with pytest.raises(RuntimeError, match="kind setter"):
        GLYPH_OPERATIONS[Glyph.EN](node, factors)

    assert node.EPI == 0.2
    assert node.epi_kind == "seed"


@pytest.mark.parametrize(
    ("glyph", "attribute"),
    [
        (Glyph.AL, "EPI"),
        (Glyph.IL, "dnfr"),
        (Glyph.SHA, "vf"),
        (Glyph.THOL, "d2EPI"),
        (Glyph.NAV, "dnfr"),
        (Glyph.ZHIR, "theta"),
    ],
)
def test_direct_handlers_reject_nonfinite_input_state_before_commit(glyph, attribute):
    node = _BareNode()
    setattr(node, attribute, float("nan"))
    factors = get_glyph_factors(node, glyph)

    with pytest.raises(TNFRValueError, match="state"):
        GLYPH_OPERATIONS[glyph](node, factors)

    assert math.isnan(getattr(node, attribute))
    assert node.storage == {"glyph_history": []}
