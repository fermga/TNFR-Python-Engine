"""Six finite controls for the operator-origin pulse closure audit.

Direct primary glyph handlers, strict Emission admission and constitutive
pressure refresh are distinct paths. No selector, public operator word,
trajectory or occurrence law is installed or replayed. The analytic threshold
guard is a supplied countermodel; its formal continuation is not an admission
certificate for the complete public/native runtime.
"""

from fractions import Fraction
from math import ulp

import networkx as nx
import pytest
import sympy as sp

from tnfr.config import inject_defaults
from tnfr.constants.canonical import SHA_VF_FACTOR, VAL_SCALE_FACTOR
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.errors import TNFRValueError
from tnfr.node import NodeNX
from tnfr.operators import (
    _make_scale_op,
    _op_AL,
    _op_IL,
    _op_SHA,
    resolve_runtime_operator_factors,
)
from tnfr.operators.al_sha_stage_proposals import (
    emission_epi_proposal,
    silence_capacity_proposal,
)
from tnfr.operators.preconditions.emission import validate_emission_strict
from tnfr.types import Glyph


def _graph(*, epi=0.125, capacity=1.0, pressure=-1.0):
    graph = nx.path_graph(2)
    inject_defaults(graph)
    graph[0][1]["weight"] = 1.0
    graph.graph.update(
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        EDGE_AWARE_ENABLED=False,
        EPI_LATENT_MAX=0.5,
        VF_BASAL_THRESHOLD=0.5,
        MIN_NETWORK_DEGREE_COUPLING=1,
        DNFR_WEIGHTS={"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0},
    )
    for index in graph:
        graph.nodes[index].update(
            EPI=epi if index == 0 else 0.375,
            nu_f=capacity,
            delta_nfr=pressure,
            theta=0.25,
            epi_kind="seed",
        )
    return graph


def _state(graph, index=0):
    node = NodeNX.from_graph(graph, index)
    return float(node.EPI), node.vf, node.dnfr, node.theta


def _primary(graph, glyph, handler, overrides=None):
    factors = resolve_runtime_operator_factors(overrides, glyph, graph.graph)
    handler(NodeNX.from_graph(graph, 0), factors)


def test_shared_primary_maps_keep_their_exact_channel_contracts():
    cases = (
        (Glyph.AL, _op_AL, {"AL_boost": 0.25}, (0.375, 1.0, -1.0, 0.25)),
        (Glyph.IL, _op_IL, {"IL_dnfr_factor": 0.5}, (0.125, 1.0, -0.5, 0.25)),
        (Glyph.SHA, _op_SHA, {"SHA_vf_factor": 0.5}, (0.125, 0.5, -1.0, 0.25)),
    )
    for glyph, handler, factors, expected in cases:
        graph = _graph()
        neighbor = _state(graph, 1)
        support = tuple(graph.edges(data="weight"))
        _primary(graph, glyph, handler, factors)
        assert _state(graph) == expected
        assert _state(graph, 1) == neighbor
        assert tuple(graph.edges(data="weight")) == support
    # Silence does not freeze form while both retained factors remain nonzero.
    assert expected[1] * expected[2] == -0.5


def test_capacity_return_requires_compensating_regeneration():
    attenuation, restoration, initial = sp.symbols("s e nu", positive=True)
    residual = initial * attenuation * restoration - initial
    assert sp.solve(residual, restoration) == [1 / attenuation]
    assert sp.expand(residual.subs(restoration, 1)) == initial * attenuation - initial
    delta = sp.symbols("delta", positive=True)
    assert sp.expand((1 + delta) * (1 - delta) - 1) == -(delta**2)

    graph = _graph()
    held = _state(graph)[::2], _state(graph)[3]
    _primary(graph, Glyph.VAL, _make_scale_op(Glyph.VAL))
    assert _state(graph)[1] == VAL_SCALE_FACTOR
    _primary(graph, Glyph.SHA, _op_SHA)
    expected = float(
        Fraction.from_float(VAL_SCALE_FACTOR) * Fraction.from_float(SHA_VF_FACTOR)
    )
    assert _state(graph)[1] == expected < 1.0
    assert (_state(graph)[::2], _state(graph)[3]) == held

    # Exact real strict attenuation does not imply strict floating decrease at
    # every positive subnormal. This is an admitted represented plateau.
    smallest = ulp(0.0)
    before, after = silence_capacity_proposal(smallest, 0.75)
    assert before == after == smallest > 0.0
    assert Fraction.from_float(smallest) * Fraction(3, 4) < Fraction.from_float(after)


def test_positive_emission_and_pressure_contraction_do_not_close_full_state():
    boost, contraction, magnitude = sp.symbols("b r magnitude", positive=True)
    integrated_flow = sp.symbols("integrated_flow", real=True)
    assert sp.solve(integrated_flow + boost, integrated_flow) == [-boost]
    # With held pressure between events, nonzero pressure return requires the
    # total pressure multiplier to equal one, independently of the form balance.
    assert sp.solve(-magnitude * contraction + magnitude, contraction) == [1]

    graph = _graph()
    before, after = emission_epi_proposal(graph.graph, 0.125, 0.25)
    assert after - before == 0.25
    for pressure in (-1.0, 1.0):
        source = _graph(pressure=pressure)
        _primary(source, Glyph.IL, _op_IL, {"IL_dnfr_factor": 0.5})
        assert _state(source)[2] == pressure / 2
        assert _state(source)[2] * pressure > 0
        assert _state(source)[0:2] == (0.125, 1.0)

    # A positive request need not be a positive realized increment at a rail;
    # saturation supplies neither negative return flow nor pressure restoration.
    before, after = emission_epi_proposal(graph.graph, 1.0, 0.25)
    assert before == after == 1.0


def test_declared_guard_has_repeated_excursions_with_growing_waits():
    n = sp.symbols("n", integer=True, nonnegative=True)
    count = sp.symbols("N", integer=True, positive=True)
    lower, boost = sp.Rational(1, 8), sp.Rational(1, 4)
    upper = lower + boost
    capacity, pressure = 2 ** (-n), -(2 ** (-n))
    wait = sp.Rational(1, 4) * 4**n
    elapsed = (4**count - 1) / 12

    assert upper == sp.Rational(3, 8)
    assert sp.simplify(upper + capacity * pressure * wait - lower) == 0
    assert sp.simplify(capacity.subs(n, n + 1) - capacity / 2) == 0
    assert sp.simplify(pressure.subs(n, n + 1) - pressure / 2) == 0
    assert sp.simplify(wait.subs(n, n + 1) - 4 * wait) == 0
    assert sp.simplify(sp.summation(wait, (n, 0, count - 1)) - elapsed) == 0
    assert sp.limit(elapsed, count, sp.oo) == sp.oo  # No Zeno accumulation.
    assert tuple(elapsed.subs(count, k) for k in (1, 2, 3)) == (
        sp.Rational(1, 4),
        sp.Rational(5, 4),
        sp.Rational(21, 4),
    )
    # Every excursion has the same endpoints, while both hidden factors and
    # the slope change. Equality of observed form is insufficient for return.
    assert sp.simplify(
        (capacity * pressure).subs(n, n + 1) / (capacity * pressure)
    ) == sp.Rational(1, 4)
    assert sp.simplify(capacity.subs(n, n + 1) / capacity) == sp.Rational(1, 2)


def test_actual_emission_admission_exits_after_capacity_attenuation():
    # These are three independently prepared threshold states, not a runtime
    # trajectory or a claim of complete public-word admission.
    for capacity in (1.0, 0.5):
        graph = _graph(capacity=capacity)
        before = _state(graph)
        assert validate_emission_strict(graph, 0) is None
        assert _state(graph) == before
    graph = _graph(capacity=0.25)
    before = _state(graph)
    with pytest.raises(TNFRValueError, match="frequency too low"):
        validate_emission_strict(graph, 0)
    assert _state(graph) == before
    assert before[0] < graph.graph["EPI_LATENT_MAX"]
    assert graph.degree(0) >= graph.graph["MIN_NETWORK_DEGREE_COUPLING"]


def test_pressure_refresh_erases_only_the_stored_primary_coherence_change():
    graph = _graph()
    structural = tuple((_state(graph, i)[:2], _state(graph, i)[3]) for i in graph)
    support = tuple(graph.edges(data="weight"))
    # Explicit pure-EPI completion: each node has one unit-conductance neighbor.
    default_compute_delta_nfr(graph)
    assert tuple(_state(graph, i)[2] for i in graph) == (0.25, -0.25)
    _primary(graph, Glyph.IL, _op_IL, {"IL_dnfr_factor": 0.5})
    assert tuple(_state(graph, i)[2] for i in graph) == (0.125, -0.25)
    assert (
        tuple((_state(graph, i)[:2], _state(graph, i)[3]) for i in graph) == structural
    )
    default_compute_delta_nfr(graph)
    assert tuple(_state(graph, i)[2] for i in graph) == (0.25, -0.25)
    assert (
        tuple((_state(graph, i)[:2], _state(graph, i)[3]) for i in graph) == structural
    )
    assert tuple(graph.edges(data="weight")) == support
    # Public/staged Coherence can also align primitive phase. This control is
    # only about the shared primary pressure write and the declared completion.
