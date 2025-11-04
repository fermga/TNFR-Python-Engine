"""Property tests validating node initialisation against graph configuration."""

from __future__ import annotations

import copy
import math
from typing import Mapping

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st

from tnfr.constants import THETA_KEY, VF_KEY
from tnfr.initialization import init_node_attrs

from .strategies import PROPERTY_TEST_SETTINGS, prepare_network

def _resolve_uniform_bounds(params: Mapping[str, float | None]) -> tuple[float, float]:
    """Return the effective ``νf`` bounds used for uniform initialisation."""

    vf_min_lim = float(params["VF_MIN"])
    vf_max_lim = float(params["VF_MAX"])
    raw_min = params.get("INIT_VF_MIN")
    raw_max = params.get("INIT_VF_MAX")
    vf_uniform_min = vf_min_lim if raw_min is None else float(raw_min)
    vf_uniform_max = vf_max_lim if raw_max is None else float(raw_max)
    if vf_uniform_min > vf_uniform_max:
        vf_uniform_min, vf_uniform_max = vf_uniform_max, vf_uniform_min
    vf_uniform_min = max(vf_uniform_min, vf_min_lim)
    vf_uniform_max = min(vf_uniform_max, vf_max_lim)
    # After clamping to VF_MIN/VF_MAX, ensure min <= max
    if vf_uniform_min > vf_uniform_max:
        # Collapse to VF_MIN when the requested range is entirely below the limit
        vf_uniform_min = vf_uniform_max = vf_min_lim
    return vf_uniform_min, vf_uniform_max

def _collect_node_attrs(graph, *, override: bool) -> dict[int, dict[str, float]]:
    """Initialise ``graph`` and capture the core node attributes."""

    init_node_attrs(graph, override=override)
    return {
        node: {
            "EPI": data["EPI"],
            THETA_KEY: data[THETA_KEY],
            VF_KEY: data[VF_KEY],
            "Si": data["Si"],
        }
        for node, data in graph.nodes(data=True)
    }

def _assert_node_attributes(
    attrs: Mapping[int, Mapping[str, float]],
    config: Mapping[str, object],
    uniform_bounds: tuple[float, float],
) -> None:
    """Assert that node attributes respect ``config`` selections."""

    theta_min, theta_max = sorted(
        (float(config["INIT_THETA_MIN"]), float(config["INIT_THETA_MAX"]))
    )
    random_phase = bool(config["INIT_RANDOM_PHASE"])
    vf_mode = str(config["INIT_VF_MODE"]).lower()
    vf_min_lim = float(config["VF_MIN"])
    vf_max_lim = float(config["VF_MAX"])
    si_min = float(config["INIT_SI_MIN"])
    si_max = float(config["INIT_SI_MAX"])
    epi_val = float(config["INIT_EPI_VALUE"])
    uniform_min, uniform_max = uniform_bounds

    for values in attrs.values():
        epi = values["EPI"]
        theta = values[THETA_KEY]
        vf = values[VF_KEY]
        si = values["Si"]

        assert math.isclose(epi, epi_val, rel_tol=0.0, abs_tol=1e-12)
        assert si_min <= si <= si_max
        assert theta_min <= theta <= theta_max
        if not random_phase:
            assert math.isclose(theta, 0.0, abs_tol=1e-12)
        assert vf_min_lim <= vf <= vf_max_lim
        if vf_mode == "uniform":
            assert uniform_min <= vf <= uniform_max

def _bounded_float(min_value: float, max_value: float) -> st.SearchStrategy[float]:
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )

@st.composite
def _init_parameter_sets(draw) -> dict[str, object]:
    random_phase = draw(st.booleans())
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    vf_min_lim = draw(_bounded_float(-0.25, 0.9))
    vf_max_upper = min(vf_min_lim + 1.0, 1.5)
    vf_max_lim = draw(_bounded_float(vf_min_lim + 0.05, vf_max_upper))

    uniform_bounds = draw(
        st.tuples(
            st.one_of(st.none(), _bounded_float(vf_min_lim - 0.5, vf_max_lim + 0.5)),
            st.one_of(st.none(), _bounded_float(vf_min_lim - 0.5, vf_max_lim + 0.5)),
        )
    )
    uniform_min, uniform_max = uniform_bounds
    if (
        uniform_min is not None
        and uniform_max is not None
        and draw(st.booleans())
    ):
        uniform_min, uniform_max = uniform_max, uniform_min

    if random_phase:
        theta_min = draw(_bounded_float(-2 * math.pi, 2 * math.pi))
        theta_max = draw(_bounded_float(-2 * math.pi, 2 * math.pi))
    else:
        theta_min = draw(_bounded_float(-2 * math.pi, 0.0))
        theta_max = draw(_bounded_float(0.0, 2 * math.pi))
        if draw(st.booleans()):
            theta_min, theta_max = theta_max, theta_min

    si_min = draw(_bounded_float(0.0, 0.95))
    si_max = draw(_bounded_float(si_min, min(si_min + 0.5, 1.5)))

    epi_val = draw(_bounded_float(-1.0, 1.0))

    return {
        "RANDOM_SEED": seed,
        "INIT_RANDOM_PHASE": random_phase,
        "INIT_THETA_MIN": theta_min,
        "INIT_THETA_MAX": theta_max,
        "VF_MIN": vf_min_lim,
        "VF_MAX": vf_max_lim,
        "INIT_VF_MIN": uniform_min,
        "INIT_VF_MAX": uniform_max,
        "INIT_VF_MODE": draw(st.sampled_from(("uniform", "normal"))),
        "INIT_VF_CLAMP_TO_LIMITS": draw(st.booleans()),
        "INIT_SI_MIN": si_min,
        "INIT_SI_MAX": si_max,
        "INIT_EPI_VALUE": epi_val,
    }

@PROPERTY_TEST_SETTINGS
@given(data=st.data(), init_config=_init_parameter_sets())
def test_init_node_attrs_respects_graph_configuration(data, init_config) -> None:
    graph = data.draw(
        prepare_network(min_nodes=2, max_nodes=6, connected=False, init_nodes=False),
        label="graph",
    )
    graph.graph.update(init_config)

    uniform_bounds = _resolve_uniform_bounds(graph.graph)

    def clone():
        return copy.deepcopy(graph)

    collected = {
        override: _collect_node_attrs(clone(), override=override)
        for override in (True, False)
    }
    repeated = {
        override: _collect_node_attrs(clone(), override=override)
        for override in (True, False)
    }

    for override in (True, False):
        assert collected[override] == repeated[override]
        _assert_node_attributes(
            collected[override],
            graph.graph,
            uniform_bounds,
        )
