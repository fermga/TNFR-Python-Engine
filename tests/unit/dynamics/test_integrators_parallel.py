import copy
import math

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import get_aliases
from tnfr.dynamics import integrators as integrators_mod
from tnfr.dynamics.integrators import update_epi_via_nodal_equation

ALIAS_EPI = get_aliases("EPI")
ALIAS_DEPI = get_aliases("DEPI")
ALIAS_D2EPI = get_aliases("D2EPI")


def _build_sample_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.graph.update(
        {
            "DT": 0.15,
            "DT_MIN": 0.05,
            "INTEGRATOR_METHOD": "euler",
            "_t": 0.0,
            "GAMMA": {"type": "harmonic", "beta": 0.42, "omega": 0.31, "phi": 0.17},
        }
    )
    for idx in range(6):
        G.add_node(
            idx,
            VF=1.0 + 0.05 * idx,
            DNFR=0.2 * math.cos(0.3 * idx),
            EPI=0.5 * idx,
            DEPI=0.1 * math.sin(0.2 * idx),
            THETA=0.4 * idx,
        )
    return G


def _snapshot(G: nx.DiGraph) -> dict[int, tuple[float, float, float]]:
    return {
        node: (
            get_attr(data, ALIAS_EPI, 0.0),
            get_attr(data, ALIAS_DEPI, 0.0),
            get_attr(data, ALIAS_D2EPI, 0.0),
        )
        for node, data in G.nodes(data=True)
    }


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_parallel_integrator_matches_serial(method: str) -> None:
    base = _build_sample_graph()
    serial = copy.deepcopy(base)
    parallel = copy.deepcopy(base)

    kwargs = {"dt": 0.15, "t": 0.0, "method": method}
    update_epi_via_nodal_equation(serial, **kwargs, n_jobs=None)
    update_epi_via_nodal_equation(parallel, **kwargs, n_jobs=3)

    assert parallel.graph["_t"] == pytest.approx(serial.graph["_t"])
    serial_snapshot = _snapshot(serial)
    parallel_snapshot = _snapshot(parallel)
    for node in serial_snapshot:
        assert parallel_snapshot[node] == pytest.approx(serial_snapshot[node])


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_parallel_fallback_without_numpy(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _build_sample_graph()
    base.graph["INTEGRATOR_METHOD"] = method

    np_missing_calls = 0

    def fake_get_numpy() -> None:
        nonlocal np_missing_calls
        np_missing_calls += 1
        return None

    monkeypatch.setattr(integrators_mod, "get_numpy", fake_get_numpy)

    serial = copy.deepcopy(base)
    parallel = copy.deepcopy(base)

    kwargs = {"dt": base.graph["DT"], "t": base.graph["_t"], "method": method}
    update_epi_via_nodal_equation(serial, **kwargs, n_jobs=None)
    update_epi_via_nodal_equation(parallel, **kwargs, n_jobs=3)

    assert np_missing_calls > 0
    serial_snapshot = _snapshot(serial)
    parallel_snapshot = _snapshot(parallel)
    for node in serial_snapshot:
        assert parallel_snapshot[node] == pytest.approx(serial_snapshot[node])
