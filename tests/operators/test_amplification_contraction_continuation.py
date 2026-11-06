"""Finite actual RA/NUL words, live renewal and a closed-population contrast.

The prospective protocol is in artifacts/amplification-contraction-20261103.
All sources are supplied two-node states with actual IL preparation. No old
producer, acquired-source helper, continuous flow or assigned past is used.
"""

import math

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.config import inject_defaults
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SOURCE_GLYPH,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.errors import TNFRValueError
from tnfr.operators import Coherence, Contraction, Coupling, Resonance
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    TWO_PHASE_JACOBI,
    execute_coupling_stage,
    execute_neighbor_stage,
    execute_pointwise_stage,
)
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils import angle_diff
from tnfr.utils._structural_signature import structural_proof_signature

C, D = "receiver", "donor"
_ABS = 2e-14


def _raw(graph, node, aliases):
    return get_attr(graph.nodes[node], aliases, strict=True, conv=lambda value: value)


def _form(graph, node):
    return require_finite_real_scalar_epi(_raw(graph, node, ALIAS_EPI))


def _primary(graph, node):
    return (
        _form(graph, node),
        _raw(graph, node, ALIAS_VF),
        _raw(graph, node, ALIAS_DNFR),
        _raw(graph, node, ALIAS_THETA),
    )


def _checkpoint(graph):
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


def _assert_result(result, glyph, targets):
    assert result.glyph == glyph
    assert result.schedule == TWO_PHASE_JACOBI
    assert result.nodes_processed == len(targets)


def _pointwise(graph, operator, targets, **kwargs):
    result = execute_pointwise_stage(
        graph, operator, targets, require_exact_operator=True, **kwargs
    )
    _assert_result(result, operator.glyph.value, targets)
    return result


def _resonate(graph, targets=(C,)):
    result = execute_neighbor_stage(graph, Resonance(), targets)
    _assert_result(result, "RA", targets)
    return result


def _couple(graph, targets=(C,)):
    result = execute_coupling_stage(graph, Coupling(), targets, preserve_relations=True)
    _assert_result(result, "UM", targets)
    return result


def _source(
    *,
    form=3 / 8,
    capacity=1 / 2,
    pressure=3 / 16,
    phase=1 / 4,
    donor_form=3 / 4,
    donor_capacity=1.0,
    donor_pressure=0.0,
    amplification=1 / 8,
):
    graph = nx.Graph()
    for node, x, vf, p, theta in (
        (C, form, capacity, pressure, phase),
        (D, donor_form, donor_capacity, donor_pressure, 0.0),
    ):
        graph.add_node(
            node,
            **{
                ALIAS_EPI[0]: x,
                ALIAS_VF[0]: vf,
                ALIAS_DNFR[0]: 2 * p,
                ALIAS_THETA[0]: theta,
                ALIAS_EPI_KIND[0]: "form",
                "Si": 0.0,
            },
        )
    graph.add_edge(C, D, weight=1.0, role="supplied")
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=73,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        IL_PRECONDITIONS={"warn_zero_dnfr": False, "warn_isolated": False},
        EDGE_AWARE_ENABLED=True,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        UM_FUNCTIONAL_LINKS=False,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
    )
    factors = graph.graph["GLYPH_FACTORS"]
    # Derive the reciprocal from this supplied NUL scale, not the old default.
    factors.pop("NUL_densification_factor", None)
    factors.update(
        RA_epi_diff=0.5,
        RA_vf_amplification=amplification,
        RA_phase_coupling=0.5,
        NUL_scale=0.5,
        UM_theta_push=0.5,
        UM_vf_sync=0.25,
        UM_dnfr_reduction=0.25,
        IL_dnfr_factor=0.5,
    )
    _pointwise(graph, Coherence(), (C, D), phase_locking_coefficient=0.0)
    assert _primary(graph, C) == (form, capacity, pressure, phase)
    assert _primary(graph, D) == (
        donor_form,
        donor_capacity,
        donor_pressure,
        0.0,
    )
    _assert_context(graph, C, ("IL",))
    _assert_context(graph, D, ("IL",))
    _assert_boundary(graph)
    return graph


def _assert_context(graph, node, expected):
    state = read_grammar_state(graph.nodes[node])
    window = graph.graph["GLYPH_HYSTERESIS_WINDOW"]
    assert window > 0
    assert state.history == tuple(expected)[-window:]
    assert state.operator_step == len(expected)
    assert state.u2_debt == 0
    assert state.prior_coherence is True
    assert _raw(graph, node, ALIAS_SOURCE_GLYPH) == expected[-1]


def _assert_boundary(graph):
    assert tuple(graph) == (C, D)
    assert tuple(graph.edges) == ((C, D),)
    assert graph.edges[C, D] == {"weight": 1.0, "role": "supplied"}
    assert graph.graph["_t"] == 0.0
    assert abs(angle_diff(_primary(graph, C)[3], _primary(graph, D)[3])) < math.pi / 2
    for node in (C, D):
        assert _raw(graph, node, ALIAS_EPI_KIND) == "form"
        # These named jumps have not acquired an elapsed-time form secant.
        assert "epi_time_history" not in graph.nodes[node]


def _return(graph, targets=(C,)):
    """Execute the four fixed stages and retain their actual prefix states."""
    observations = []
    _resonate(graph, targets)
    _assert_boundary(graph)
    observations.append(tuple(_primary(graph, node) for node in (C, D)))
    _pointwise(graph, Contraction(), targets)
    _assert_boundary(graph)
    observations.append(tuple(_primary(graph, node) for node in (C, D)))
    _couple(graph, targets)
    _assert_boundary(graph)
    observations.append(tuple(_primary(graph, node) for node in (C, D)))
    _pointwise(graph, Coherence(), targets, phase_locking_coefficient=0.5)
    _assert_boundary(graph)
    observations.append(tuple(_primary(graph, node) for node in (C, D)))
    return tuple(observations)


@pytest.mark.parametrize(
    "amplification,pressure,rounds,q",
    ((1 / 8, 1 / 8, 2, 9 / 16), (1.0, 3 / 8, 1, 1.0), (3 / 2, 1 / 8, 2, 5 / 4)),
)
def test_actual_ra_nul_capacity_classification_does_not_cancel_pressure(
    amplification, pressure, rounds, q
):
    graph = _source(
        form=1 / 4,
        capacity=1.0,
        pressure=pressure,
        amplification=amplification,
    )
    donor_before = _primary(graph, D)
    expected_history = ["IL"]

    for count in range(1, rounds + 1):
        _resonate(graph)
        assert _form(graph, C) == 1 / 2
        assert _primary(graph, C)[2] == pressure * 2 ** (count - 1)
        _pointwise(graph, Contraction(), (C,))
        expected_history.extend(("RA", "NUL"))
        x, capacity, p, phase = _primary(graph, C)
        assert (x, capacity, p) == (1 / 4, q**count, pressure * 2**count)
        assert phase == pytest.approx((1 / 4) / 2**count, rel=0, abs=_ABS)
        assert _primary(graph, D) == donor_before
        _assert_context(graph, C, expected_history)
        _assert_context(graph, D, ("IL",))
        _assert_boundary(graph)
        record = graph.graph["nul_densification_log"][-1]
        assert record["node"] == C
        assert record["densification_factor"] == 2.0
        assert record["dnfr_after"] == p
        assert record["binary64_inverse_product_residual"] == 0.0

    if q == 1.0:
        # Neutral capacity retained neither pressure nor the old own context.
        assert _primary(graph, C)[1:3] == (1.0, 3 / 4)
        before = _checkpoint(graph)
        with pytest.raises(TNFRValueError, match="controlled dissonance"):
            _resonate(graph)
        assert _checkpoint(graph) == before
        _assert_context(graph, C, ("IL", "RA", "NUL"))


@pytest.mark.parametrize("initial_gap", (0.0, 1 / 4))
def test_actual_live_renewal_retains_full_numerical_rows_and_own_memory(initial_gap):
    graph = _source(phase=initial_gap)
    history = ["IL"]
    pressure = 3 / 16
    prefix_gap = initial_gap
    expected = (
        (9 / 32, 59 / 128, initial_gap / 4, initial_gap / 8),
        (33 / 128, 3641 / 8192, 5 * initial_gap / 32, 9 * initial_gap / 64),
    )

    for form, capacity, target_phase, donor_phase in expected:
        before = _primary(graph, C)
        rows = _return(graph)
        ra, nul, um, final = (row[0] for row in rows)
        assert ra[:3] == ((before[0] + 3 / 4) / 2, 9 * before[1] / 8, before[2])
        assert nul[:3] == (form, 9 * before[1] / 16, 2 * before[2])
        assert um[:2] == (form, capacity)
        # UM stabilization consumes its own final gap, before IL locks again.
        expected_um_pressure = 2 * pressure * (3 / 4 + prefix_gap / (16 * math.pi))
        assert um[2] == pytest.approx(expected_um_pressure, rel=0, abs=_ABS)
        pressure *= 3 / 4 + prefix_gap / (16 * math.pi)
        assert final[:2] == (form, capacity)
        assert final[2] == pytest.approx(pressure, rel=0, abs=_ABS)
        assert final[3] == pytest.approx(target_phase, rel=0, abs=_ABS)
        assert rows[-1][1][:3] == (3 / 4, 1.0, 0.0)
        assert rows[-1][1][3] == pytest.approx(donor_phase, rel=0, abs=_ABS)
        assert abs(angle_diff(final[3], rows[-1][1][3])) == pytest.approx(
            prefix_gap / 8, rel=0, abs=_ABS
        )
        history.extend(("RA", "NUL", "UM", "IL"))
        _assert_context(graph, C, history)
        _assert_context(graph, D, ("IL",))
        prefix_gap /= 8

    assert 3 / 8 < _primary(graph, C)[1] < 1 / 2
    assert 0 < _primary(graph, C)[2] < 3 / 16
    if initial_gap:
        assert _primary(graph, D)[3] > 0  # Secondary participation is real.


def test_actual_donor_contraction_changes_live_renewal_without_cached_reservoir():
    observed = {}
    for contract_donor in (False, True):
        graph = _source()
        donor_history = ["IL"]
        if contract_donor:
            _pointwise(graph, Contraction(), (D,))
            donor_history.append("NUL")
            assert _primary(graph, D) == (3 / 8, 1 / 2, 0.0, 0.0)
        _return(graph)
        observed[contract_donor] = (_primary(graph, C), _primary(graph, D))
        _assert_context(graph, C, ("IL", "RA", "NUL", "UM", "IL"))
        _assert_context(graph, D, donor_history)
        _assert_boundary(graph)

    intact, contracted = observed[False], observed[True]
    assert intact[0][:2] == (9 / 32, 59 / 128)
    assert contracted[0][:2] == (3 / 16, 43 / 128)
    assert contracted[0][0] - intact[0][0] == -3 / 32
    assert contracted[0][1] - intact[0][1] == -1 / 8
    assert contracted[0][2:] == pytest.approx(intact[0][2:], rel=0, abs=_ABS)
    assert contracted[1][:3] == (3 / 8, 1 / 2, 0.0)
    assert intact[1][:3] == (3 / 4, 1.0, 0.0)
    assert contracted[1][3] == pytest.approx(intact[1][3], rel=0, abs=_ABS)


def test_closed_equal_pair_has_no_external_capacity_renewal():
    graph = _source(
        form=1 / 2,
        capacity=1 / 2,
        pressure=3 / 16,
        phase=0.0,
        donor_form=1 / 2,
        donor_capacity=1 / 2,
        donor_pressure=3 / 16,
    )
    history = ["IL"]
    expected = ((1 / 4, 9 / 32, 9 / 64), (1 / 8, 81 / 512, 27 / 256))
    for form, capacity, pressure in expected:
        rows = _return(graph, (C, D))
        for index, node in enumerate((C, D)):
            # Simultaneous UM reads the already depleted equal capacities.
            assert rows[2][index][1] == rows[1][index][1] == capacity
            assert _primary(graph, node) == (form, capacity, pressure, 0.0)
        history.extend(("RA", "NUL", "UM", "IL"))
        _assert_context(graph, C, history)
        _assert_context(graph, D, history)
        _assert_boundary(graph)
    assert _primary(graph, C)[1] == (1 / 2) * (9 / 16) ** 2
